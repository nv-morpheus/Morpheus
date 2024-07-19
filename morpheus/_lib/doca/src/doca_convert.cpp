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
#include "morpheus/objects/dev_mem_info.hpp" // for DevMemInfo
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"
#include "morpheus/utilities/matx_util.hpp"            // for MatxUtil

#include <boost/fiber/context.hpp>
#include <cuda_runtime.h>
#include <cudf/concatenate.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
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
    morpheus::doca::packet_data_buffer& data,
    morpheus::doca::packet_data_buffer& sizes,
    rmm::cuda_stream_view stream)
{
    data.shrink_to_fit();
    CHECK(sizes.elements == data.elements);

    auto offsets_buffer = morpheus::doca::sizes_to_offsets(sizes.elements, sizes.data<uint32_t>(), stream);

    const auto offset_count = sizes.elements+1;
    const auto offset_buff_size = (offset_count)*sizeof(int32_t);

    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32), offset_count, std::move(offsets_buffer), std::move(rmm::device_buffer(0, stream)), 0);



    return cudf::make_strings_column(data.elements,
                                     std::move(offsets_col),
                                     std::move(data.buffer),
                                     0,
                                     {});
}

std::unique_ptr<cudf::column> make_ip_col(morpheus::doca::packet_data_buffer& data, rmm::cuda_stream_view stream)
{
    // cudf doesn't support uint32, need to cast to int64
    data.shrink_to_fit();
    const morpheus::TensorIndex num_packets = static_cast<morpheus::TensorIndex>(data.elements);

    auto src_type = morpheus::DType::create<uint32_t>();
    auto dst_type =  morpheus::DType(morpheus::TypeId::INT64);
    auto src_buffer = std::make_shared<rmm::device_buffer>(std::move(data.buffer));
    auto dev_mem_info =  morpheus::DevMemInfo(src_buffer, src_type, {num_packets}, {1});

    auto ip_int64_buff =  morpheus::MatxUtil::cast(dev_mem_info, dst_type.type_id());
    
    auto src_ip_int_col = std::make_unique<cudf::column>(cudf::data_type(dst_type.cudf_type_id()), 
                                                         num_packets,
                                                         std::move(*ip_int64_buff), 
                                                         std::move(rmm::device_buffer(0, stream)), 
                                                         0);

    return cudf::strings::integers_to_ipv4(src_ip_int_col->view());
}


namespace morpheus {

DocaConvertStage::DocaConvertStage(std::chrono::milliseconds max_time_delta,
                                   std::size_t sizes_buffer_size,
                                   std::size_t header_buffer_size,
                                   std::size_t payload_buffer_size) :
  base_t(base_t::op_factory_from_sub_fn(build())),
  m_max_time_delta{max_time_delta},
  m_sizes_buffer_size{sizes_buffer_size},
  m_header_buffer_size{header_buffer_size},
  m_payload_buffer_size{payload_buffer_size}
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp              = rmm::cuda_stream_view(reinterpret_cast<cudaStream_t>(m_stream));
    m_fixed_pld_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&m_fixed_pld_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        m_fixed_pld_size_list_cpu[idx] = MAX_PKT_SIZE;
    MRC_CHECK_CUDA(cudaMemcpy(m_fixed_pld_size_list, m_fixed_pld_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault));

    m_fixed_hdr_size_list_cpu = (uint32_t*)calloc(MAX_PKT_RECEIVE, sizeof(uint32_t));
    cudaMalloc((void**)&m_fixed_hdr_size_list, MAX_PKT_RECEIVE * sizeof(uint32_t));
    for (int idx = 0; idx < MAX_PKT_RECEIVE; idx++)
        m_fixed_hdr_size_list_cpu[idx] = IP_ADDR_STRING_LEN;
    MRC_CHECK_CUDA(cudaMemcpy(m_fixed_hdr_size_list, m_fixed_hdr_size_list_cpu, MAX_PKT_RECEIVE * sizeof(uint32_t), cudaMemcpyDefault));

    auto mr = rmm::mr::get_current_device_resource();
    m_header_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(m_header_buffer_size, m_stream_cpp, mr);
    m_header_sizes_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(m_sizes_buffer_size, m_stream_cpp, mr);

    m_payload_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(m_payload_buffer_size, m_stream_cpp, mr);
    m_payload_sizes_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(m_sizes_buffer_size, m_stream_cpp, mr);

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
                if (!m_header_buffer->empty()) {
                    std::cerr << "flushing buffer prior to shutdown" << std::endl << std::flush;
                    send_buffered_data(output);
                }
                output.on_completed();
            }));
    };
}

void log_time(const std::string& item,
              const std::chrono::time_point<std::chrono::steady_clock> t1,
              const std::chrono::time_point<std::chrono::steady_clock> t2)
{
    // TODO: Remove before merging
    auto diff = std::chrono::duration_cast<std::chrono::duration<float>>(t2-t1);
    std::cerr << item << " : " << std::fixed << diff.count() << "\n";
}

void DocaConvertStage::on_raw_packet_message(rxcpp::subscriber<source_type_t>& output, sink_type_t raw_msg)
{
    auto packet_count      = raw_msg->count();
    auto max_size          = raw_msg->get_max_size();
    auto pkt_addr_list     = raw_msg->get_pkt_addr_list();
    auto pkt_hdr_size_list = raw_msg->get_pkt_hdr_size_list();
    auto pkt_pld_size_list = raw_msg->get_pkt_pld_size_list();
    auto queue_idx         = raw_msg->get_queue_idx();


    // LOG(WARNING) << "New RawPacketMessage with " << packet_count << " packets from queue id " << queue_idx;

    auto t1 = std::chrono::steady_clock::now();
    const auto payload_buff_size = doca::gather_sizes(packet_count, pkt_pld_size_list,  m_stream_cpp);

    auto t2 = std::chrono::steady_clock::now();
    log_time("gather_sizes", t1, t2);

    const uint32_t header_buff_size = packet_count * sizeof(uint32_t);
    const auto sizes_buff_size = packet_count * sizeof(uint32_t);

    bool buffers_full = (header_buff_size > m_header_buffer->available_bytes() ||
                         sizes_buff_size > m_header_sizes_buffer->available_bytes() ||
                         payload_buff_size > m_payload_buffer->available_bytes() ||
                         sizes_buff_size > m_payload_sizes_buffer->available_bytes());

    auto cur_time = std::chrono::steady_clock::now();
    auto time_since_last_emit = cur_time - m_last_emit;
    bool buffer_time_expired = (time_since_last_emit >= m_max_time_delta);

    // TODO: If the buffers are not full, but buffer_time_expired, we should include the current message in the output.
    // TODO: The current logic for calculating expiration time doesn't account for the possibility that we could have a
    // large burst of incoming data and then not receive anymore data for quite some time. In this case we could have
    // data in the buffer long past the m_max_time_delta
    if (buffers_full || buffer_time_expired)
    {
        // Build a MessageMeta emit it, and reset the buffers.
        // There is a possibility that the buffers are empty, but either we crossed a time-delta before receiving new
        // messages or the allocated buffers are too small for the incoming
        // RawPacketMessage, when this is the case we should log a warning and allocate a larger buffer
        bool buffer_has_data = !m_header_buffer->empty();
        if (buffer_has_data)
        {
            send_buffered_data(output);
        }

        // In the case where existing empty buffers were too small for the data we still need to re-allocate
        if (buffer_has_data || buffers_full)
        {
            auto header_size = get_alloc_size(m_header_buffer_size, header_buff_size, "header");
            m_header_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(header_size, m_stream_cpp);

            auto payload_size = get_alloc_size(m_payload_buffer_size, payload_buff_size, "payload");
            m_payload_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(payload_size, m_stream_cpp);

            auto sizes_size = get_alloc_size(m_sizes_buffer_size, sizes_buff_size, "sizes");
            m_header_sizes_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(sizes_size, m_stream_cpp);
            m_payload_sizes_buffer = std::make_unique<morpheus::doca::packet_data_buffer>(sizes_size, m_stream_cpp);
        }
    }

    // this should never be true
    DCHECK(header_buff_size <= m_header_buffer->available_bytes() &&
           sizes_buff_size <= m_header_sizes_buffer->available_bytes() &&
           payload_buff_size <= m_payload_buffer->available_bytes() &&
           sizes_buff_size <= m_payload_sizes_buffer->available_bytes());


#if ENABLE_TIMERS == 1
    const auto t0 = now_ns();
#endif

    auto t3 = std::chrono::steady_clock::now();
    // gather header data
    doca::gather_header(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_header_buffer->current_location<uint32_t>(), m_stream_cpp);

    auto t4 = std::chrono::steady_clock::now();

    log_time("gather_header", t3, t4);

    // gather payload data
    doca::gather_payload(
        packet_count, pkt_addr_list, pkt_hdr_size_list, pkt_pld_size_list, m_payload_buffer->current_location(), m_stream_cpp);

    auto t5 = std::chrono::steady_clock::now();

    log_time("gather_payload", t4, t5);
    cudaStreamSynchronize(m_stream_cpp);

    auto t6 = std::chrono::steady_clock::now();
    log_time("sync", t5, t6);

#if ENABLE_TIMERS == 1
    const auto t1 = now_ns();
#endif

    m_header_buffer->advance(header_buff_size, packet_count);
    m_payload_buffer->advance(payload_buff_size, packet_count);

    MRC_CHECK_CUDA(cudaMemcpy(m_header_sizes_buffer->current_location(), m_fixed_hdr_size_list, sizes_buff_size, cudaMemcpyDeviceToDevice));
    m_header_sizes_buffer->advance(sizes_buff_size, packet_count);

    MRC_CHECK_CUDA(cudaMemcpy(m_payload_sizes_buffer->current_location(), pkt_pld_size_list, sizes_buff_size, cudaMemcpyDeviceToDevice));
    m_payload_sizes_buffer->advance(sizes_buff_size, packet_count);

    auto t7 = std::chrono::steady_clock::now();
    log_time("cudaMemcpy", t6, t7);
    std::cerr << "\n\n\n";

#if ENABLE_TIMERS == 1
    const auto t2 = now_ns();
#endif

}

void DocaConvertStage::send_buffered_data(rxcpp::subscriber<source_type_t>& output)
{
    auto cudf_t1 = std::chrono::steady_clock::now();
    const auto num_packets = m_payload_buffer->elements;
    CHECK(num_packets == m_header_buffer->elements);
    
    auto src_ip_col = make_ip_col(*m_header_buffer, m_stream_cpp);
    auto payload_col = make_string_col(*m_payload_buffer, *m_payload_sizes_buffer, m_stream_cpp);

    std::vector<std::unique_ptr<cudf::column>> gathered_columns;
    gathered_columns.emplace_back(std::move(src_ip_col));
    gathered_columns.emplace_back(std::move(payload_col));

    auto gathered_table = std::make_unique<cudf::table>(std::move(gathered_columns));

    auto gathered_metadata = cudf::io::table_metadata();
    gathered_metadata.schema_info.emplace_back("src_ip");
    gathered_metadata.schema_info.emplace_back("data");

    auto gathered_table_w_metadata =
        cudf::io::table_with_metadata{std::move(gathered_table), std::move(gathered_metadata)};

    auto meta = MessageMeta::create_from_cpp(std::move(gathered_table_w_metadata), 0);
    output.on_next(std::move(meta));

    auto now = std::chrono::steady_clock::now();
    m_last_emit = now;


    // auto delta = now - m_last_emit;
    // std::chrono::duration<float> deltaf = std::chrono::duration_cast<std::chrono::duration<float>>(delta);
    // float count = deltaf.count();
    // auto packet_rate = (num_packets / count);

    log_time("cudf_time", cudf_t1, now);

    // const float bytes = m_payload_buffer->cur_offset_bytes + m_header_buffer->cur_offset_bytes;
    // const float mb = bytes / (1024 * 1024);
    // std::cerr << "Convert Packets: " << num_packets
    //           << "\nDelta: " << deltaf.count()
    //           << "s\nPackets/s : " << packet_rate
    //           << "\nMB/s: " << mb << "\n\n";
}

std::shared_ptr<mrc::segment::Object<DocaConvertStage>> DocaConvertStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    std::string const& name,
    std::chrono::milliseconds max_time_delta,
    std::size_t sizes_buffer_size,
    std::size_t header_buffer_size,
    std::size_t payload_buffer_size)
{
    return builder.construct_object<DocaConvertStage>(
        name, max_time_delta, sizes_buffer_size, header_buffer_size, payload_buffer_size);
}

}  // namespace morpheus
