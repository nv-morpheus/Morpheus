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
#include "morpheus/doca/doca_convert_stage.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"
#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"         // for DType
#include "morpheus/utilities/matx_util.hpp"   // for MatxUtil

#include <boost/fiber/context.hpp>
#include <boost/fiber/fiber.hpp>
#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/convert/convert_ipv4.hpp>
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

std::size_t get_alloc_size(std::size_t default_size, uint32_t incoming_size, const std::string& buffer_name)
{
    if (incoming_size > default_size)
    {
        LOG(WARNING) << "RawPacketMessage requires a " << buffer_name << " buffer of size " << incoming_size
                     << " bytes, but the default allocation size is only " << default_size << " allocating "
                     << incoming_size;

        return incoming_size;
    }

    return default_size;
}

std::unique_ptr<cudf::column> make_string_col(morpheus::doca::packet_data_buffer& packet_buffer)
{
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                      packet_buffer.m_num_packets + 1,
                                                      std::move(*packet_buffer.m_payload_offsets_buffer),
                                                      std::move(rmm::device_buffer(0, packet_buffer.m_stream)),
                                                      0);

    return cudf::make_strings_column(packet_buffer.m_num_packets, std::move(offsets_col), std::move(*packet_buffer.m_payload_buffer), 0, {});
}

std::unique_ptr<cudf::column> make_ip_col(morpheus::doca::packet_data_buffer& packet_buffer)
{
    // cudf doesn't support uint32, need to cast to int64
    const morpheus::TensorIndex num_packets = static_cast<morpheus::TensorIndex>(packet_buffer.m_num_packets);

    auto src_type     = morpheus::DType::create<uint32_t>();
    auto dst_type     = morpheus::DType(morpheus::TypeId::INT64);
    auto dev_mem_info = morpheus::DevMemInfo(packet_buffer.m_header_buffer, src_type, {num_packets}, {1});

    auto ip_int64_buff = morpheus::MatxUtil::cast(dev_mem_info, dst_type.type_id());

    auto src_ip_int_col = std::make_unique<cudf::column>(cudf::data_type(dst_type.cudf_type_id()),
                                                         num_packets,
                                                         std::move(*ip_int64_buff),
                                                         std::move(rmm::device_buffer(0, packet_buffer.m_stream)),
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
  m_payload_buffer_size{payload_buffer_size},
  m_buffer_channel{std::make_shared<mrc::BufferedChannel<doca::packet_data_buffer>>(payload_buffer_size)}
{
    cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
    m_stream_cpp = rmm::cuda_stream_view(m_stream);
}

DocaConvertStage::~DocaConvertStage()
{
    cudaStreamDestroy(m_stream);
}

DocaConvertStage::subscribe_fn_t DocaConvertStage::build()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        
        auto buffer_reader_fiber = boost::fibers::fiber([this, &output]()
        {
            this->buffer_reader(output);
        });

        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output, &buffer_reader_fiber](sink_type_t x) {
                this->on_raw_packet_message(output, x);
            },
            [&](std::exception_ptr error_ptr) {
                output.on_error(error_ptr);
            },
            [&]() {
                m_buffer_channel->close_channel();
                buffer_reader_fiber.join()
            }));
    };
}

void DocaConvertStage::on_raw_packet_message(sink_type_t raw_msg)
{
    auto packet_count      = raw_msg->count();
    auto max_size          = raw_msg->get_max_size();
    auto pkt_addr_list     = raw_msg->get_pkt_addr_list();
    auto pkt_hdr_size_list = raw_msg->get_pkt_hdr_size_list();
    auto pkt_pld_size_list = raw_msg->get_pkt_pld_size_list();
    auto queue_idx         = raw_msg->get_queue_idx();

    const auto payload_buff_size = doca::gather_sizes(packet_count, pkt_pld_size_list, m_stream_cpp);

    const uint32_t header_buff_size = packet_count * sizeof(uint32_t);

    // The offsets buffer needs to start with a 0, and end with the offset of the last element, and is always one
    // element longer than the number of packets
    const auto sizes_buff_size = (packet_count + 1) * sizeof(uint32_t);

    auto packet_buffer = doca::packet_data_buffer(packet_count, header_buff_size, payload_buff_size, sizes_buff_size, m_stream_cpp);
    
    morpheus::doca::sizes_to_offsets(packet_buffer.m_num_packets, 
                                     pkt_pld_size_list,
                                     *packet_buffer.m_payload_offsets_buffer,
                                     packet_buffer.m_stream);

    // gather header data
    doca::gather_header(packet_count,
                        pkt_addr_list,
                        pkt_hdr_size_list,
                        pkt_pld_size_list,
                        static_cast<uint32_t*>(packet_buffer.m_header_buffer->data()),
                        m_stream_cpp);

    doca::gather_payload(packet_count,
                         pkt_addr_list,
                         pkt_hdr_size_list,
                         pkt_pld_size_list,
                         static_cast<int32_t*>(packet_buffer.m_payload_offsets_buffer->data()),
                         static_cast<uint8_t*>(packet_buffer.m_payload_buffer->data()),
                         m_stream_cpp);

    cudaStreamSynchronize(m_stream_cpp);

    m_buffer_channel->await_write(std::move(packet_buffer));
}

void DocaConvertStage::buffer_reader(rxcpp::subscriber<source_type_t>& output)
{
    std::vector<doca::packet_data_buffer> packets;

    while (!m_buffer_channel->is_channel_closed())
    {
        auto poll_start = std::chrono::steady_clock::now();
        auto poll_end = poll_start + m_max_time_delta;
        while (std::chrono::steady_clock::now() < poll_end && !m_buffer_channel->is_channel_closed())
        {
            doca::packet_data_buffer packet_buffer;
            auto status = m_buffer_channel->await_read_until(packet_buffer, poll_end);

            if (status == mrc::channel::Status::success)
            {
                packets.emplace_back(std::move(packet_buffer));
            }
        }

        if (!packets.empty())
        {
            packet_data_buffer combined_data(std::move(packets));
            send_buffered_data(output, std::move(combined_data));
            packets.clear();
        }
    }
}

void DocaConvertStage::send_buffered_data(rxcpp::subscriber<source_type_t>& output, doca::packet_data_buffer&& packet_buffer)
{
    auto src_ip_col  = make_ip_col(packet_buffer);
    auto payload_col = make_string_col(packet_buffer);

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
