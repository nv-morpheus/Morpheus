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

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/doca_kernels.hpp"
#include "morpheus/doca/doca_stages.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <boost/fiber/context.hpp>
#include <cuda_runtime.h>
#include <cudf/concatenate.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <generic/rte_byteorder.h>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rxcpp/rx.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
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
#define ENABLE_TIMERS 0

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

#define DEBUG_GET_TIMESTAMP(ts) clock_gettime(CLOCK_REALTIME, (ts))

std::unique_ptr<packet_data_buffer> make_packer_data_buffer(
    std::size_t size,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr= rmm::mr::get_current_device_resource())
{
    auto buffer = rmm::device_buffer(size, stream, mr);
    return std::make_unique<packet_data_buffer>(std::move(buffer), 0, 0);
}

std::size_t get_alloc_size(std::size_t default_size, uint32_t incoming_size, const std::string& buffer_name)
{
    if (incoming_size > default_size)
    {
        LOG(WARNING) << "RawPacketMessage requires a " << buffer_name << " buffer of size " << incoming_size
                     << " bytes, but the default allocation size is only " << default_size
                     << " allocating " << incoming_size;

        return incoming_size;
    }

    return default_size;
}

std::unique_ptr<cudf::column> make_string_col(
    packet_data_buffer& data,
    packet_data_buffer& sizes,
    rmm::cuda_stream_view stream)
{
    data.shrink_to_fit();
    sizes.shrink_to_fit();

    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32), sizes.elements, std::move(sizes.buffer), std::move(rmm::device_buffer(0, stream)), 0);

    return cudf::make_strings_column(data.elements,
                                     std::move(offsets_col),
                                     std::move(data.buffer),
                                     0,
                                     {});
}


namespace morpheus {

DocaConvertStage::DocaConvertStage() :
  base_t(base_t::op_factory_from_sub_fn(build()))
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp              = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(m_stream));
    m_fixed_pld_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&m_fixed_pld_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        m_fixed_pld_size_list_cpu[idx] = MAX_PKT_SIZE;
    cudaMemcpy(m_fixed_pld_size_list, m_fixed_pld_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault);

    m_fixed_hdr_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&m_fixed_hdr_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        m_fixed_hdr_size_list_cpu[idx] = IP_ADDR_STRING_LEN;
    cudaMemcpy(m_fixed_hdr_size_list, m_fixed_hdr_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault);

    auto mr = rmm::mr::get_current_device_resource();
    m_header_buffer = make_packer_data_buffer(m_header_buffer_size, m_stream_cpp, mr);
    m_header_offsets_buffer = make_packer_data_buffer(m_sizes_buffer_size, m_stream_cpp, mr);

    m_payload_buffer = make_packer_data_buffer(m_payload_buffer_size, m_stream_cpp, mr);
    m_payload_offsets_buffer = make_packer_data_buffer(m_sizes_buffer_size, m_stream_cpp, mr);

}

DocaConvertStage::~DocaConvertStage()
{
    free(m_fixed_pld_size_list_cpu);
    cudaFree(m_fixed_pld_size_list);
    free(m_fixed_hdr_size_list_cpu);
    cudaFree(m_fixed_hdr_size_list);
    cudaStreamDestroy(m_stream);
}

DocaConvertStage::subscribe_fn_t DocaConvertStage::build()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        this->m_last_emit = std::chrono::steady_clock::now();
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                this->on_raw_packet_message(output, x);
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                output.on_completed();
            }));
    };
}

// DocaConvertStage::source_type_t DocaConvertStage::on_data(sink_type_t x)
// {
//     if constexpr (std::is_same_v<sink_type_t, std::shared_ptr<RawPacketMessage>>)
//     {
//         return this->on_raw_packet_message(x);
//     }
//     // sink_type_t not supported
//     else
//     {
//         std::string error_msg{"DocaConvertStage receives unsupported input type: " + std::string(typeid(x).name())};
//         LOG(ERROR) << error_msg;
//         throw std::runtime_error(error_msg);
//     }
// }

void DocaConvertStage::on_raw_packet_message(rxcpp::subscriber<source_type_t>& output, sink_type_t raw_msg)
{
    auto packet_count      = raw_msg->count();
    auto max_size          = raw_msg->get_max_size();
    auto pkt_addr_list     = raw_msg->get_pkt_addr_list();
    auto pkt_hdr_size_list = raw_msg->get_pkt_hdr_size_list();
    auto pkt_pld_size_list = raw_msg->get_pkt_pld_size_list();
    auto queue_idx         = raw_msg->get_queue_idx();

    // std::vector<uint32_t> header_sizes(packet_count);
    // cudaMemcpy(pkt_hdr_size_list, header_sizes.data(), packet_count*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // std::cerr << "header sizes: ";
    // for (auto hs: header_sizes){
    //     std::cerr << hs << ", ";
    // }
    // std::cerr << "\n";

    // std::vector<uint32_t> payload_sizes(packet_count);
    // cudaMemcpy(pkt_pld_size_list, payload_sizes.data(), packet_count*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // std::cerr << "payload sizes: ";
    // for (auto ps: payload_sizes){
    //     std::cerr << ps << ", ";
    // }
    // std::cerr << "\n";

    // LOG(WARNING) << "New RawPacketMessage with " << packet_count << " packets from queue id " << queue_idx;

    const auto [header_buff_size, payload_buff_size] = doca::gather_sizes(
        packet_count, m_fixed_hdr_size_list, m_fixed_pld_size_list,  m_stream_cpp);

    // both m_fixed_hdr_size_list and m_fixed_pld_size_list should be the same size in bytes
    auto sizes_buff_size = packet_count * sizeof(uint32_t);

    std::vector<uint32_t> host_buff(packet_count);
    MRC_CHECK_CUDA(cudaMemcpy(host_buff.data(), m_fixed_pld_size_list, sizes_buff_size, cudaMemcpyDeviceToHost));
    std::cerr << "\n************\nHost Sizes(" << packet_count <<"):\n";
    for (std::size_t i = 0; i < host_buff.size(); ++i)
    {
        std::cerr << "\t" << i << " : " << host_buff[i] << "\n";
    }

    std::cerr << "***************\n\n" << std::flush;
    


    bool buffers_full = (header_buff_size > m_header_buffer->available_bytes() ||
                         sizes_buff_size > m_header_offsets_buffer->available_bytes() ||
                         payload_buff_size > m_payload_buffer->available_bytes() ||
                         sizes_buff_size > m_payload_offsets_buffer->available_bytes());

    auto cur_time = std::chrono::steady_clock::now();
    auto time_since_last_emit = cur_time - m_last_emit;
    bool buffer_time_expired = (time_since_last_emit >= m_max_time_delta);

    // TODO: If the buffers are not full, but buffer_time_expired, we should include the current message in the output.
    if (buffers_full || buffer_time_expired)
    {
        // Build a MessageMeta emit it, and reset the buffers.
        // There is a possibility that the buffers are empty, but either we crossed a time-delta before receiving new
        // messages or the allocated buffers are too small for the incoming
        // RawPacketMessage, when this is the case we should log a warning and allocate a larger buffer
        if (!m_header_buffer->empty())
        {
            auto header_col = make_string_col(*m_header_buffer, *m_header_offsets_buffer, m_stream_cpp);
            auto payload_col = make_string_col(*m_payload_buffer, *m_payload_offsets_buffer, m_stream_cpp);

            std::vector<std::unique_ptr<cudf::column>> gathered_columns;
            gathered_columns.emplace_back(std::move(header_col));
            gathered_columns.emplace_back(std::move(payload_col));

            auto gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

            auto gathered_metadata = cudf::io::table_metadata();
            gathered_metadata.schema_info.emplace_back("src_ip");
            gathered_metadata.schema_info.emplace_back("data");

            auto gathered_table_w_metadata =
                cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

            auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);
            output.on_next(std::move(meta));

            m_last_emit = std::chrono::steady_clock::now();
        }

        if (buffers_full) 
        {
            auto header_size = get_alloc_size(m_header_buffer_size, header_buff_size, "header");
            m_header_buffer = make_packer_data_buffer(header_size, m_stream_cpp);

            auto payload_size = get_alloc_size(m_payload_buffer_size, payload_buff_size, "payload");
            m_payload_buffer = make_packer_data_buffer(payload_size, m_stream_cpp);

            auto sizes_size = get_alloc_size(m_sizes_buffer_size, sizes_buff_size, "sizes");
            m_header_offsets_buffer = make_packer_data_buffer(sizes_size, m_stream_cpp);
            m_payload_offsets_buffer = make_packer_data_buffer(sizes_size, m_stream_cpp);
        }
    }

    // this should never be true
    DCHECK(header_buff_size <= m_header_buffer->available_bytes() &&
           sizes_buff_size <= m_header_offsets_buffer->available_bytes() &&
           payload_buff_size <= m_payload_buffer->available_bytes() &&
           sizes_buff_size <= m_payload_offsets_buffer->available_bytes());


#if ENABLE_TIMERS == 1
    const auto t0 = now_ns();
#endif
    // gather header data
    doca::gather_header(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_header_buffer->current_location(), m_stream_cpp);

    // gather payload data
    doca::gather_payload(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_payload_buffer->current_location(), m_stream_cpp);

    cudaStreamSynchronize(m_stream_cpp);


#if ENABLE_TIMERS == 1
    const auto t1 = now_ns();
#endif

    m_header_buffer->advance(header_buff_size, packet_count);
    m_payload_buffer->advance(payload_buff_size, packet_count);

    doca::sizes_to_offsets(packet_count, pkt_hdr_size_list, m_header_offsets_buffer->current_location<uint32_t>(), m_stream_cpp);
    doca::sizes_to_offsets(packet_count, pkt_pld_size_list, m_payload_offsets_buffer->current_location<uint32_t>(), m_stream_cpp);
    cudaStreamSynchronize(m_stream_cpp);

    const auto offset_count = packet_count+1;
    const auto offset_buff_size = (offset_count)*sizeof(uint32_t);

    std::vector<uint32_t> host_off_buff(offset_count);
    MRC_CHECK_CUDA(cudaMemcpy(host_off_buff.data(), m_payload_offsets_buffer->current_location(), offset_buff_size, cudaMemcpyDeviceToHost));
    std::cerr << "\n************\nHost Offsets(" << offset_count <<"):\n";
    for (std::size_t i = 0; i < host_off_buff.size(); ++i)
    {
        std::cerr << "\t" << i << " : " << host_off_buff[i] << "\n";
    }

    std::cerr << "***************\n\n" << std::flush;

    m_header_offsets_buffer->advance(offset_buff_size, offset_count);
    m_payload_offsets_buffer->advance(offset_buff_size, offset_count);


#if ENABLE_TIMERS == 1
    const auto t2 = now_ns();
#endif

//     m_gathered_rows += payload_col->size();
//     std::vector<std::unique_ptr<cudf::column>> gathered_columns;
//     gathered_columns.emplace_back(std::move(header_src_ip_col));
//     gathered_columns.emplace_back(std::move(payload_col));

//     // After this point buffers can be reused -> copies actual packets' data

//     m_gathered_tables.emplace_back(std::move(std::make_unique<cudf::table>(std::move(gathered_columns))));

//     cudaStreamSynchronize(m_stream_cpp);

// #if ENABLE_TIMERS == 1
//     const auto t3 = now_ns();
// #endif

//     if (m_gathered_rows >= m_rows_per_df) {
//         auto gathered_metadata = cudf::io::table_metadata();
//         gathered_metadata.schema_info.emplace_back("src_ip");
//         gathered_metadata.schema_info.emplace_back("data");

//         std::vector<cudf::table_view> table_views;
//         for (auto& tbl: m_gathered_tables) {
//             table_views.emplace_back(tbl->view());
//         }

//         auto combined_table = cudf::concatenate(table_views, m_stream_cpp);


//         auto gathered_table_w_metadata =
//             cudf::io::table_with_metadata{std::move(combined_table), std::move(gathered_metadata)};

//     #if ENABLE_TIMERS == 1
//         const auto t4 = now_ns();
//     #endif
//         auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);

//     #if ENABLE_TIMERS == 1
//         const auto t5 = now_ns();
//     #endif
//         cudaStreamSynchronize(m_stream_cpp);

//     #if ENABLE_TIMERS == 1
//         const auto t6 = now_ns();

//         LOG(WARNING) << "Queue " << queue_idx << " packets " << packet_count << " header column " << t1 - t0
//                     << " payload column " << t2 - t1 << " gather columns " << t3 - t2 << " gather metadata " << t4 - t3
//                     << " create_from_cpp " << t5 - t4 << " stream sync " << t6 - t5 << std::endl;
//     #endif

//         m_gathered_tables.clear();
//         m_gathered_rows = 0;

//         output.on_next(std::move(meta));

//     }
}

std::shared_ptr<mrc::segment::Object<DocaConvertStage>> DocaConvertStageInterfaceProxy::init(
    mrc::segment::Builder& builder, std::string const& name)
{
    return builder.construct_object<DocaConvertStage>(name);
}

}  // namespace morpheus
