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

#include "morpheus/doca/doca_kernels.hpp"
#include "morpheus/doca/doca_stages.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <boost/fiber/context.hpp>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <generic/rte_byteorder.h>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rxcpp/rx.hpp>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

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

static uint64_t now_ns()
{
    struct timespec t;
    if (clock_gettime(CLOCK_REALTIME, &t) != 0)
        return 0;
    return (uint64_t)t.tv_nsec + (uint64_t)t.tv_sec * 1000 * 1000 * 1000;
}

#define debug_get_timestamp(ts) clock_gettime(CLOCK_REALTIME, (ts))

namespace morpheus {

DocaConvertStage::DocaConvertStage(bool const& split_hdr) :
  base_t(rxcpp::operators::map([this](sink_type_t x) {
      return this->on_data(std::move(x));
  })),
  m_split_hdr(split_hdr)
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(m_stream));

    // Protocol?
    // Total payload size?

    // assemble metadata

    // if (split_hdr)
}

DocaConvertStage::~DocaConvertStage()
{
    cudaStreamDestroy(m_stream);
}

DocaConvertStage::source_type_t DocaConvertStage::on_data(sink_type_t x)
{
    if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<RawPacketMessage>>)
    {
        return this->on_raw_packet_message(x);
    }
    // sink_type_t not supported
    else
    {
        std::string error_msg{"DocaConvertStage receives unsupported input type: " + std::string(typeid(x).name())};
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

DocaConvertStage::source_type_t DocaConvertStage::on_raw_packet_message(sink_type_t raw_msg)
{
    auto packet_count      = raw_msg->count();
    auto max_size          = raw_msg->get_max_size();
    auto pkt_addr_list     = raw_msg->get_pkt_addr_list();
    auto pkt_hdr_size_list = raw_msg->get_pkt_hdr_size_list();
    auto pkt_pld_size_list = raw_msg->get_pkt_pld_size_list();
    auto queue_idx         = raw_msg->get_queue_idx();
    DocaConvertStage::source_type_t output;

    LOG(WARNING) << "New RawPacketMessage with " << packet_count << " packets from queue id " << queue_idx;

    // gather header data
    auto header_col =
        doca::gather_header(packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_stream_cpp);

    // gather payload data
    auto payload_col =
        doca::gather_payload(packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_stream_cpp);

    // const auto gather_payload_stop = now_ns();

    std::vector<std::unique_ptr<cudf::column>> gathered_columns;
    gathered_columns.emplace_back(std::move(header_col));
    gathered_columns.emplace_back(std::move(payload_col));

    // After this point buffers can be reused -> copies actual packets' data
    auto gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

    // const auto gather_table_meta = now_ns();

    auto gathered_metadata = cudf::io::table_metadata();
    gathered_metadata.schema_info.emplace_back("header");
    gathered_metadata.schema_info.emplace_back("data");

    auto gathered_table_w_metadata =
        cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

    // const auto create_message_cpp = now_ns();

    auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);

    // const auto gather_meta_stop = now_ns();

    cudaStreamSynchronize(m_stream_cpp);

    return std::move(meta);
}

#if 0

DocaConvertStage::subscriber_fn_t DocaConvertStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {
        struct packets_info* pkt_ptr;
        int sem_idx          = 0;
        cudaStream_t rstream = nullptr;
        cudaStream_t pstream = nullptr;
        cudf::table_view* fixed_width_inputs_table_view[MAX_SEM_X_QUEUE];

        std::vector<rmm::device_uvector<char>> payload_buffer_d;
        std::vector<rmm::device_uvector<int32_t>> payload_sizes_d;
        std::vector<rmm::device_uvector<int64_t>> src_mac_out_d;
        std::vector<rmm::device_uvector<int64_t>> dst_mac_out_d;
        std::vector<rmm::device_uvector<int64_t>> src_ip_out_d;
        std::vector<rmm::device_uvector<int64_t>> dst_ip_out_d;
        std::vector<rmm::device_uvector<uint16_t>> src_port_out_d;
        std::vector<rmm::device_uvector<uint16_t>> dst_port_out_d;
        std::vector<rmm::device_uvector<int32_t>> tcp_flags_out_d;
        std::vector<rmm::device_uvector<int32_t>> ether_type_out_d;
        std::vector<rmm::device_uvector<int32_t>> next_proto_id_out_d;
        std::vector<rmm::device_uvector<uint32_t>> timestamp_out_d;

        int thread_idx = mrc::runnable::Context::get_runtime_context().rank();

        if (thread_idx >= MAX_QUEUE)
        {
            MORPHEUS_FAIL("More CPU threads than allowed queues");
        }

        payload_buffer_d.reserve(MAX_SEM_X_QUEUE);
        payload_sizes_d.reserve(MAX_SEM_X_QUEUE);
        src_mac_out_d.reserve(MAX_SEM_X_QUEUE);
        dst_mac_out_d.reserve(MAX_SEM_X_QUEUE);
        src_ip_out_d.reserve(MAX_SEM_X_QUEUE);
        dst_ip_out_d.reserve(MAX_SEM_X_QUEUE);
        src_port_out_d.reserve(MAX_SEM_X_QUEUE);
        dst_port_out_d.reserve(MAX_SEM_X_QUEUE);
        tcp_flags_out_d.reserve(MAX_SEM_X_QUEUE);
        ether_type_out_d.reserve(MAX_SEM_X_QUEUE);
        next_proto_id_out_d.reserve(MAX_SEM_X_QUEUE);
        timestamp_out_d.reserve(MAX_SEM_X_QUEUE);

        // Dedicated CUDA stream for the receiver kernel
        cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&pstream, cudaStreamNonBlocking);
        auto pstream_cpp = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(pstream));
        mrc::Unwinder ensure_cleanup([rstream, pstream]() {
            // Ensure that the stream gets cleaned up even if we error
            cudaStreamDestroy(rstream);
            cudaStreamDestroy(pstream);
        });

        for (int idxs = 0; idxs < MAX_SEM_X_QUEUE; idxs++)
        {
            payload_buffer_d.push_back(rmm::device_uvector<char>(MAX_PKT_RECEIVE * MAX_PKT_SIZE, pstream_cpp));
            payload_sizes_d.push_back(rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE, pstream_cpp));
            src_mac_out_d.push_back(rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE, pstream_cpp));
            dst_mac_out_d.push_back(rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE, pstream_cpp));
            src_ip_out_d.push_back(rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE, pstream_cpp));
            dst_ip_out_d.push_back(rmm::device_uvector<int64_t>(MAX_PKT_RECEIVE, pstream_cpp));
            src_port_out_d.push_back(rmm::device_uvector<uint16_t>(MAX_PKT_RECEIVE, pstream_cpp));
            dst_port_out_d.push_back(rmm::device_uvector<uint16_t>(MAX_PKT_RECEIVE, pstream_cpp));
            tcp_flags_out_d.push_back(rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE, pstream_cpp));
            ether_type_out_d.push_back(rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE, pstream_cpp));
            next_proto_id_out_d.push_back(rmm::device_uvector<int32_t>(MAX_PKT_RECEIVE, pstream_cpp));
            timestamp_out_d.push_back(rmm::device_uvector<uint32_t>(MAX_PKT_RECEIVE, pstream_cpp));

            pkt_ptr = static_cast<struct packets_info*>(m_semaphore[thread_idx]->get_info_cpu(idxs));
            pkt_ptr->payload_buffer_out = payload_buffer_d[idxs].data();
            pkt_ptr->payload_sizes_out  = payload_sizes_d[idxs].data();
            pkt_ptr->src_mac_out        = src_mac_out_d[idxs].data();
            pkt_ptr->dst_mac_out        = dst_mac_out_d[idxs].data();
            pkt_ptr->src_ip_out         = src_ip_out_d[idxs].data();
            pkt_ptr->dst_ip_out         = dst_ip_out_d[idxs].data();
            pkt_ptr->src_port_out       = src_port_out_d[idxs].data();
            pkt_ptr->dst_port_out       = dst_port_out_d[idxs].data();
            pkt_ptr->tcp_flags_out      = tcp_flags_out_d[idxs].data();
            pkt_ptr->ether_type_out     = ether_type_out_d[idxs].data();
            pkt_ptr->next_proto_id_out  = next_proto_id_out_d[idxs].data();
            pkt_ptr->timestamp_out      = timestamp_out_d[idxs].data();

            fixed_width_inputs_table_view[idxs] = new cudf::table_view(std::vector<cudf::column_view>{
                cudf::column_view(cudf::device_span<const int64_t>(src_mac_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int64_t>(dst_mac_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int64_t>(src_ip_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int64_t>(dst_ip_out_d[idxs])),
                cudf::column_view(cudf::device_span<const uint16_t>(src_port_out_d[idxs])),
                cudf::column_view(cudf::device_span<const uint16_t>(dst_port_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int32_t>(tcp_flags_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int32_t>(ether_type_out_d[idxs])),
                cudf::column_view(cudf::device_span<const int32_t>(next_proto_id_out_d[idxs])),
                cudf::column_view(cudf::device_span<const uint32_t>(timestamp_out_d[idxs])),
            });
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

                // const auto table_stop = now_ns();

                auto packet_count       = pkt_ptr->packet_count_out;
                auto payload_size_total = pkt_ptr->payload_size_total_out;

                // LOG(WARNING) << "CPU packet_count " << packet_count << " payload_size_total " << payload_size_total
                // << std::endl;

                // Should not be necessary
                if (packet_count == 0)
                    continue;

                // gather payload data
                auto payload_col = doca::gather_payload(
                    packet_count, pkt_ptr->payload_buffer_out, pkt_ptr->payload_sizes_out, pstream_cpp);

                // const auto gather_payload_stop = now_ns();

                auto iota_col = [packet_count]() {
                    using scalar_type_t = cudf::scalar_type_t<uint32_t>;
                    auto zero =
                        cudf::make_numeric_scalar(cudf::data_type(cudf::data_type{cudf::type_to_id<uint32_t>()}));
                    static_cast<scalar_type_t*>(zero.get())->set_value(0);
                    zero->set_valid_async(false);
                    return cudf::sequence(packet_count, *zero);
                }();

                // Accept the stream now?
                auto gathered_table   = cudf::gather(*fixed_width_inputs_table_view[sem_idx],
                                                   iota_col->view(),
                                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                                   pstream_cpp);
                auto gathered_columns = gathered_table->release();

                // const auto table_create_stop = now_ns();

                // post-processing for mac addresses
                auto src_mac_col = gathered_columns[0].release();
                // Accept the stream now?
                auto src_mac_str_col = morpheus::doca::integers_to_mac(src_mac_col->view(), pstream_cpp);
                gathered_columns[0].reset(src_mac_str_col.release());

                auto dst_mac_col     = gathered_columns[1].release();
                auto dst_mac_str_col = morpheus::doca::integers_to_mac(dst_mac_col->view(), pstream_cpp);
                gathered_columns[1].reset(dst_mac_str_col.release());

                // post-processing for ip addresses
                auto src_ip_col = gathered_columns[2].release();
                // Accept the stream now?
                auto src_ip_str_col = cudf::strings::integers_to_ipv4(src_ip_col->view(), pstream_cpp);
                gathered_columns[2].reset(src_ip_str_col.release());

                auto dst_ip_col = gathered_columns[3].release();
                // Accept the stream now?
                auto dst_ip_str_col = cudf::strings::integers_to_ipv4(dst_ip_col->view(), pstream_cpp);
                gathered_columns[3].reset(dst_ip_str_col.release());

                gathered_columns.emplace_back(std::move(payload_col));

                // const auto gathered_columns_stop = now_ns();

                auto gathered_metadata = cudf::io::table_metadata();
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

                // After this point buffers can be reused -> copies actual packets' data
                gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

                // const auto gather_table_meta = now_ns();

                auto gathered_table_w_metadata =
                    cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

                // const auto create_message_cpp = now_ns();

                auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);

                // Do we still need this synchronize?
                //  const auto gather_meta_stop = now_ns();

                cudaStreamSynchronize(pstream_cpp);
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
#endif

std::shared_ptr<mrc::segment::Object<DocaConvertStage>> DocaConvertStageInterfaceProxy::init(
    mrc::segment::Builder& builder, std::string const& name, bool const& split_hdr)
{
    return builder.construct_object<DocaConvertStage>(name, split_hdr);
}

}  // namespace morpheus
