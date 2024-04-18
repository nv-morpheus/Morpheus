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

#include "cuda.h"

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/doca_context.hpp"
#include "morpheus/doca/doca_kernels.hpp"
#include "morpheus/doca/doca_mem.hpp"
#include "morpheus/doca/doca_rx_pipe.hpp"
#include "morpheus/doca/doca_rx_queue.hpp"
#include "morpheus/doca/doca_semaphore.hpp"
#include "morpheus/doca/doca_stages.hpp"
#include "morpheus/messages/raw_packet.hpp"
#include "morpheus/utilities/error.hpp"

#include <boost/fiber/context.hpp>
#include <cuda_runtime.h>
#include <doca_gpunetio.h>
#include <doca_types.h>
#include <mrc/core/utils.hpp>
#include <mrc/runnable/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>
#include <stdint.h>
#include <time.h>

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

DocaSourceStage::~DocaSourceStage() = default;

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
        CUdevice cuDevice;
        CUcontext cuContext;

        cudaSetDevice(0);  // Need to rely on GPU 0
        cudaFree(0);
        cuDeviceGet(&cuDevice, 0);
        cuCtxCreate(&cuContext, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDevice);
        cuCtxPushCurrent(cuContext);

        struct packets_info* pkt_ptr;
        int sem_idx[MAX_QUEUE] = {0};
        cudaStream_t rstream = nullptr;
        int thread_idx = mrc::runnable::Context::get_runtime_context().rank();

        // Add per queue
        auto pkt_addr_unique = std::make_unique<morpheus::doca::DocaMem<uintptr_t>>(
            m_context, MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE, DOCA_GPU_MEM_TYPE_GPU);
        auto pkt_hdr_size_unique = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(
            m_context, MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE, DOCA_GPU_MEM_TYPE_GPU);
        auto pkt_pld_size_unique = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(
            m_context, MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE, DOCA_GPU_MEM_TYPE_GPU);

        if (thread_idx > 1)
        {
            MORPHEUS_FAIL("Only 1 CPU threads is allowed to run the DOCA Source Stage");
        }

        // Dedicated CUDA stream for the receiver kernel
        cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);
        mrc::Unwinder ensure_cleanup([rstream]() {
            // Ensure that the stream gets cleaned up even if we error
            cudaStreamDestroy(rstream);
        });

        for (int queue_idx = 0; queue_idx < MAX_QUEUE; queue_idx++) {
            for (int idxs = 0; idxs < MAX_SEM_X_QUEUE; idxs++) {
                pkt_ptr = static_cast<struct packets_info*>(m_semaphore[queue_idx]->get_info_cpu(idxs));
                pkt_ptr->pkt_addr     = pkt_addr_unique->gpu_ptr() + (MAX_PKT_RECEIVE * idxs);
                pkt_ptr->pkt_hdr_size = pkt_hdr_size_unique->gpu_ptr() + (MAX_PKT_RECEIVE * idxs);
                pkt_ptr->pkt_pld_size = pkt_pld_size_unique->gpu_ptr() + (MAX_PKT_RECEIVE * idxs);
            }
        }

        auto exit_condition =
            std::make_unique<morpheus::doca::DocaMem<uint32_t>>(m_context, 1, DOCA_GPU_MEM_TYPE_GPU_CPU);
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

            // const auto start_kernel = now_ns();

            // Assume MAX_QUEUE == 2
            morpheus::doca::packet_receive_kernel(m_rxq[0]->rxq_info_gpu(), m_rxq[1]->rxq_info_gpu(),
                                                  m_semaphore[0]->gpu_ptr(), m_semaphore[1]->gpu_ptr(),
                                                  sem_idx[0], sem_idx[1],
                                                  (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP) ? true : false,
                                                  exit_condition->gpu_ptr(),
                                                  rstream);
            cudaStreamSynchronize(rstream);

            // const auto end_kernel = now_ns();

            for (int queue_idx = 0; queue_idx < MAX_QUEUE; queue_idx++) {
                if (m_semaphore[queue_idx]->is_ready(sem_idx[queue_idx])) {
                    // const auto start_sem = now_ns();
                    // LOG(WARNING) << "CPU READY sem " << sem_idx[queue_idx] << " queue " << thread_idx << std::endl;

                    pkt_ptr = static_cast<struct packets_info*>(m_semaphore[queue_idx]->get_info_cpu(sem_idx[queue_idx]));

                    // Should not be necessary
                    if (pkt_ptr->packet_count_out == 0)
                        continue;

                    // LOG(WARNING) << "pkts " << pkt_ptr->packet_count_out << " MAX_PKT_SIZE " << MAX_PKT_SIZE;
                    // std::endl;

                    auto meta = RawPacketMessage::create_from_cpp(pkt_ptr->packet_count_out,
                                                                MAX_PKT_SIZE,
                                                                pkt_ptr->pkt_addr,
                                                                pkt_ptr->pkt_hdr_size,
                                                                pkt_ptr->pkt_pld_size,
                                                                true,
                                                                queue_idx);

                    // const auto create_msg = now_ns();

                    output.on_next(std::move(meta));

                    m_semaphore[queue_idx]->set_free(sem_idx[queue_idx]);
                    sem_idx[queue_idx] = (sem_idx[queue_idx] + 1) % MAX_SEM_X_QUEUE;

                    // const auto end = now_ns();

                    // LOG(WARNING) << "Queue " << queue_idx
                    //             << " packets " << pkt_ptr->packet_count_out
                    //             << " kernel time ns " << end_kernel - start_kernel
                    //             << " Sem + msg ns " << create_msg - start_sem
                    //             << " End ns " << end - create_msg
                    //             << std::endl;
                }
            }
        }

        cancel_thread.join();

        output.on_completed();

        cuCtxPopCurrent(&cuContext);
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
