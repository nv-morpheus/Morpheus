/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/doca/doca_source.hpp"

#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/doca_rx_pipe.hpp"
#include "morpheus/doca/doca_rx_queue.hpp"
#include "morpheus/doca/doca_semaphore.hpp"
#include "morpheus/doca/doca_source_kernels.hpp"
#include "morpheus/utilities/error.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/table/table.hpp>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rte_byteorder.h>

#include <iostream>
#include <memory>
#include <stdexcept>

#define BE_IPV4_ADDR(a, b, c, d) (RTE_BE32((a << 24) + (b << 16) + (c << 8) + d)) /* Big endian conversion */

std::optional<uint32_t> ip_to_int(std::string const& ip_address)
{
    if (ip_address.empty())
    {
        return 0;
    }

    uint8_t a, b, c, d;
    uint32_t ret;

    ret = sscanf(ip_address.c_str(), "%hhu.%hhu.%hhu.%hhu", &a, &b, &c, &d);

    printf("%u: %u %u %u %u\n", ret, a, b, c, d);

    if (ret == 4)
    {
        return BE_IPV4_ADDR(a, b, c, d);
    }

    return std::nullopt;
}

#define debug_get_timestamp(ts) clock_gettime(CLOCK_REALTIME, (ts))

namespace morpheus {

DocaSourceStage::DocaSourceStage(std::string const& nic_pci_address,
                                 std::string const& gpu_pci_address,
                                 std::string const& traffic_type) :
  PythonSource(build())
{
    m_context = std::make_shared<morpheus::doca::DocaContext>(nic_pci_address, gpu_pci_address);

    m_traffic_type = DOCA_TRAFFIC_TYPE_UDP;
    if (traffic_type == "tcp")
        m_traffic_type = DOCA_TRAFFIC_TYPE_TCP;

    m_rxq.reserve(MAX_QUEUE);
    m_semaphore.reserve(MAX_QUEUE);
    for (int idx = 0; idx < MAX_QUEUE; idx++)
    {
        m_rxq.push_back(std::make_shared<morpheus::doca::DocaRxQueue>(m_context));
        m_semaphore.push_back(std::make_shared<morpheus::doca::DocaSemaphore>(m_context, MAX_SEM_X_QUEUE));
    }

    m_rxpipe = std::make_shared<morpheus::doca::DocaRxPipe>(m_context, m_rxq, m_traffic_type);
}

static uint64_t now_ns()
{
    struct timespec t;
    if (clock_gettime(CLOCK_REALTIME, &t) != 0)
        return 0;
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

DocaSourceStage::subscriber_fn_t DocaSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        struct packets_info* pkt_ptr;
        int sem_idx          = 0;
        cudaStream_t rstream = nullptr;
        int thread_idx = mrc::runnable::Context::get_runtime_context().rank();

        if (thread_idx >= MAX_QUEUE)
        {
            MORPHEUS_FAIL("More CPU threads than allowed queues");
        }

        // Dedicated CUDA stream for the receiver kernel
        cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);
        mrc::Unwinder ensure_cleanup([rstream]() {
            // Ensure that the stream gets cleaned up even if we error
            cudaStreamDestroy(rstream);
        });

        for (int idxs = 0; idxs < MAX_SEM_X_QUEUE; idxs++)
        {
            pkt_ptr = static_cast<struct packets_info*>(m_semaphore[thread_idx]->get_info_cpu(idxs));
            pkt_ptr->pkt_addr = std::make_unique<morpheus::doca::DocaMem<uintptr_t>>(m_context, MAX_PKT_RECEIVE, DOCA_GPU_MEM_TYPE_GPU)->gpu_ptr();
            pkt_ptr->pkt_hdr_size = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(m_context, MAX_PKT_RECEIVE, DOCA_GPU_MEM_TYPE_GPU)->gpu_ptr();
            pkt_ptr->pkt_pld_size = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(m_context, MAX_PKT_RECEIVE, DOCA_GPU_MEM_TYPE_GPU)->gpu_ptr();
        }

        auto exit_condition = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(m_context, 1, DOCA_GPU_MEM_TYPE_GPU_CPU);
        DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) = 0;

        auto cancel_thread = std::thread([&] {
            while (output.is_subscribed()) {}
            DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) = 1;
        });

        while (output.is_subscribed())
        {
            if (DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) == 1)
            {
                output.unsubscribe();
                continue;
            }

            // printf("Launching kernel with idx0 %d idx1 %d idx2 %d\n", sem_idx[0], sem_idx[1], sem_idx[2]);
            // const auto start_kernel = now_ns();
            morpheus::doca::packet_receive_kernel(m_rxq[thread_idx]->rxq_info_gpu(),
                                                  m_semaphore[thread_idx]->gpu_ptr(),
                                                  sem_idx,
                                                  (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP) ? true : false,
                                                  exit_condition->gpu_ptr(),
                                                  rstream);
            cudaStreamSynchronize(rstream);

            if (m_semaphore[thread_idx]->is_ready(sem_idx))
            {
                // const auto start = now_ns();
                // LOG(WARNING) << "CPU READY sem " << idxs << " queue " << thread_idx << std::endl;

                pkt_ptr = static_cast<struct packets_info*>(m_semaphore[thread_idx]->get_info_cpu(sem_idx));

                // Should not be necessary
                if (pkt_ptr->packet_count_out == 0)
                    continue;

                auto meta = RawPacketMessage::create_from_cpp(pkt_ptr->packet_count_out,
                                                                MAX_PKT_SIZE,
                                                                pkt_ptr->pkt_addr,
                                                                pkt_ptr->pkt_hdr_size,
                                                                pkt_ptr->pkt_pld_size,
                                                                true,
                                                                thread_idx);

                //  const auto gather_meta_stop = now_ns();

                output.on_next(std::move(meta));

                m_semaphore[thread_idx]->set_free(sem_idx);
                sem_idx = (sem_idx + 1) % MAX_SEM_X_QUEUE;

                // const auto end = now_ns();
                // LOG(WARNING) << "Queue " << thread_idx
                //             << " packets " << packet_count
                //             << " kernel time ns " << start - start_kernel
                //             << " CPU time ns " << end - start
                //             << " table creation ns " << table_stop - start
                //             << " gather payload ns " << gather_payload_stop - table_stop
                //             << " table create ns " << table_create_stop - gather_payload_stop
                //             << " gather column ns " << gathered_columns_stop - table_create_stop
                //             << " gather meta schema ns " << gather_table_meta - gathered_columns_stop
                //             << " gather meta table ns " << create_message_cpp - gather_table_meta
                //             << " create message cpp ns " << gather_meta_stop - create_message_cpp
                //             << " final ns " << end - gather_meta_stop
                //             << std::endl;
            }
        }

        cancel_thread.join();

        output.on_completed();
    };
}

std::shared_ptr<mrc::segment::Object<DocaSourceStage>> DocaSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    std::string const& name,
    std::string const& nic_pci_address,
    std::string const& gpu_pci_address,
    std::string const& traffic_type)
{
    return builder.construct_object<DocaSourceStage>(name, nic_pci_address, gpu_pci_address, traffic_type);
}

}  // namespace morpheus
