/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doca_source.hpp"

#include "doca_context.hpp"
#include "doca_rx_pipe.hpp"
#include "doca_rx_queue.hpp"
#include "doca_semaphore.hpp"
#include "doca_source_kernels.hpp"

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

namespace morpheus {

DocaSourceStage::DocaSourceStage(std::string const& nic_pci_address, std::string const& gpu_pci_address, std::string const& traffic_type) :
  PythonSource(build())
{
    m_context = std::make_shared<morpheus::doca::DocaContext>(nic_pci_address, gpu_pci_address);

    m_traffic_type = DOCA_TRAFFIC_TYPE_UDP;
    if (traffic_type == "tcp")
        m_traffic_type = DOCA_TRAFFIC_TYPE_TCP;

    m_rxq.reserve(MAX_QUEUE);
    m_semaphore.reserve(MAX_QUEUE);
    for (int idx = 0; idx < MAX_QUEUE; idx++) {
        m_rxq.push_back(std::make_shared<morpheus::doca::DocaRxQueue>(m_context));
        m_semaphore.push_back(std::make_shared<morpheus::doca::DocaSemaphore>(m_context, MAX_SEM_X_QUEUE));
    }

    m_rxpipe    = std::make_shared<morpheus::doca::DocaRxPipe>(m_context, m_rxq, m_traffic_type);
    
    rstream = rmm::cuda_stream::cuda_stream();
}

DocaSourceStage::subscriber_fn_t DocaSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        auto payload_buffer_d     = rmm::device_uvector<char>(MAX_PKT_RECEIVE * MAX_PKT_SIZE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto payload_sizes_d      = rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto src_mac_out_d       = rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto dst_mac_out_d       = rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto src_ip_out_d        = rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto dst_ip_out_d        = rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto src_port_out_d      = rmm::device_uvector<uint16_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto dst_port_out_d      = rmm::device_uvector<uint16_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto tcp_flags_out_d     = rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto ether_type_out_d    = rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto next_proto_id_out_d = rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto timestamp_out_d     = rmm::device_uvector<uint32_t>(MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * MAX_QUEUE, rmm::cuda_stream_default);
        auto exit_condition = std::make_unique<morpheus::doca::DocaMem<uint32_t>>(m_context, 1, DOCA_GPU_MEM_TYPE_GPU_CPU);
        DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) = 0;

        // Ensure all memops on the default stream have been done
        cudaStreamSynchronize(rmm::cuda_stream_default);

        // Dedicated CUDA stream for the receiver kernel
        cudaStream_t rstream = nullptr;
        MRC_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Unwinder ensure_cleanup([rstream]() {
            // Ensure that the stream gets cleaned up even if we error
            cudaStreamDestroy(rstream);
        });

        /* Assign right pointers to each semaphore item for each queue */
        struct packets_info *pkt_ptr;
        for (int idxq = 0; idxq < MAX_QUEUE; idxq++) {
            for (int idxs = 0; idxs < MAX_SEM_X_QUEUE; idxs++) {
                pkt_ptr = static_cast<struct packets_info *>(m_semaphore[idxq]->get_info_cpu(idxs));

                pkt_ptr->payload_buffer_out   = (payload_buffer_d.data() + (MAX_PKT_RECEIVE * MAX_PKT_SIZE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * MAX_PKT_SIZE * idxs));
                pkt_ptr->payload_sizes_out    = (payload_sizes_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->src_mac_out      = (src_mac_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->dst_mac_out      = (dst_mac_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->src_ip_out       = (src_ip_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->dst_ip_out       = (dst_ip_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->src_port_out     = (src_port_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->dst_port_out     = (dst_port_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->tcp_flags_out    = (tcp_flags_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->ether_type_out   = (ether_type_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->next_proto_id_out = (next_proto_id_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
                pkt_ptr->timestamp_out    = (timestamp_out_d.data() + (MAX_PKT_RECEIVE * MAX_SEM_X_QUEUE * idxq) + (MAX_PKT_RECEIVE * idxs));
            }
        }

        auto cancel_thread = std::thread([&] {
            while (output.is_subscribed()) {}
            DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) = 1;
        });

        // Assume MAX_QUEUE == 3 queues
        morpheus::doca::packet_receive_kernel(m_rxq[0]->rxq_info_gpu(), m_rxq[1]->rxq_info_gpu(), m_rxq[2]->rxq_info_gpu(),
                                              m_semaphore[0]->gpu_ptr(), m_semaphore[1]->gpu_ptr(), m_semaphore[2]->gpu_ptr(),
                                              (m_traffic_type == DOCA_TRAFFIC_TYPE_TCP) ? true : false,
                                              exit_condition->gpu_ptr(),
                                              rstream);

        int semaphore_indexes[MAX_QUEUE] = {0};
        while (output.is_subscribed())
        {
            if (DOCA_GPUNETIO_VOLATILE(*(exit_condition->cpu_ptr())) == 1) {
                output.unsubscribe();
                continue;
            }

            for (int idxq = 0; idxq < MAX_QUEUE; idxq++) {
                if (m_semaphore[idxq]->is_ready(semaphore_indexes[idxq])) {
                    pkt_ptr = static_cast<struct packets_info *>(m_semaphore[idxq]->get_info_cpu(semaphore_indexes[idxq]));
                                auto fixed_width_inputs_table_view = cudf::table_view(std::vector<cudf::column_view>{
                                    cudf::column_view(cudf::device_span<const int64_t>(pkt_ptr->src_mac_out)),
                                    cudf::column_view(cudf::device_span<const int64_t>(pkt_ptr->dst_mac_out)),
                                    cudf::column_view(cudf::device_span<const int64_t>(pkt_ptr->src_ip_out)),
                                    cudf::column_view(cudf::device_span<const int64_t>(pkt_ptr->dst_ip_out)),
                                    cudf::column_view(cudf::device_span<const uint16_t>(pkt_ptr->src_port_out)),
                                    cudf::column_view(cudf::device_span<const uint16_t>(pkt_ptr->dst_port_out)),
                                    cudf::column_view(cudf::device_span<const int32_t>(pkt_ptr->tcp_flags_out)),
                                    cudf::column_view(cudf::device_span<const int32_t>(pkt_ptr->ether_type_out)),
                                    cudf::column_view(cudf::device_span<const int32_t>(pkt_ptr->next_proto_id_out)),
                                    cudf::column_view(cudf::device_span<const uint32_t>(pkt_ptr->timestamp_out)),
                    });

                    auto packet_count = pkt_ptr->packet_count_out;
                    auto payload_size_total = pkt_ptr->payload_size_total_out;

                    //Should not be necessary
                    if (packet_count == 0)
                        continue;

                    // gather payload data
                    auto payload_col = doca::gather_payload(packet_count, pkt_ptr->payload_buffer_out, pkt_ptr->payload_sizes_out);

                    auto iota_col = [packet_count]() {
                        using scalar_type_t = cudf::scalar_type_t<uint32_t>;
                        auto zero = cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<uint32_t>()}));
                        static_cast<scalar_type_t*>(zero.get())->set_value(0);
                        zero->set_valid_async(false);
                        return cudf::sequence(packet_count, *zero);
                    }();

                    auto gathered_table    = cudf::gather(fixed_width_inputs_table_view, iota_col->view());
                    auto gathered_metadata = cudf::io::table_metadata();
                    auto gathered_columns  = gathered_table->release();

                    // post-processing for mac addresses
                    auto src_mac_col     = gathered_columns[0].release();
                    auto src_mac_str_col = morpheus::doca::integers_to_mac(src_mac_col->view());
                    gathered_columns[0].reset(src_mac_str_col.release());

                    auto dst_mac_col     = gathered_columns[1].release();
                    auto dst_mac_str_col = morpheus::doca::integers_to_mac(dst_mac_col->view());
                    gathered_columns[1].reset(dst_mac_str_col.release());

                    // post-processing for ip addresses
                    auto src_ip_col     = gathered_columns[2].release();
                    auto src_ip_str_col = cudf::strings::integers_to_ipv4(src_ip_col->view());
                    gathered_columns[2].reset(src_ip_str_col.release());

                    auto dst_ip_col     = gathered_columns[3].release();
                    auto dst_ip_str_col = cudf::strings::integers_to_ipv4(dst_ip_col->view());
                    gathered_columns[3].reset(dst_ip_str_col.release());

                    gathered_columns.emplace_back(std::move(payload_col));

                    // assemble metadata
                    gathered_metadata.schema_info.emplace_back("src_mac");
                    gathered_metadata.schema_info.emplace_back("dst_mac");
                    gathered_metadata.schema_info.emplace_back("src_ip");
                    gathered_metadata.schema_info.emplace_back("dst_ip");
                    gathered_metadata.schema_info.emplace_back("src_port");
                    gathered_metadata.schema_info.emplace_back("dst_port");
                    gathered_metadata.schema_info.emplace_back("tcp_flags");
                    gathered_metadata.schema_info.emplace_back("ether_type");
                    gathered_metadata.schema_info.emplace_back("next_proto");
                    gathered_metadata.schema_info.emplace_back("timestamp");
                    gathered_metadata.schema_info.emplace_back("data");

                    //After this point buffers can be reused -> copies actual packets' data
                    gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

                    auto gathered_table_w_metadata =
                        cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

                    auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);

                    //Do we still need this synchronize?
                    cudaStreamSynchronize(rmm::cuda_stream_default);

                    output.on_next(std::move(meta));

                    m_semaphore[idxq]->set_free(semaphore_indexes[idxq]);
                    semaphore_indexes[idxq] = (semaphore_indexes[idxq]+1)%MAX_SEM_X_QUEUE;
                }
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
    std::string const& gpu_pci_address)
{
    return builder.construct_object<DocaSourceStage>(name, nic_pci_address, gpu_pci_address);
}

}  // namespace morpheus
