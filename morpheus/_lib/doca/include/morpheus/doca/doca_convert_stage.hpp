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

#pragma once

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/packet_data_buffer.hpp"
#include "morpheus/export.h"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <mrc/channel/buffered_channel.hpp>
#include <mrc/segment/builder.hpp>
#include <pymrc/node.hpp>

#include <chrono>
#include <memory>

namespace morpheus {

constexpr std::chrono::milliseconds DEFAULT_MAX_TIME_DELTA = std::chrono::seconds(3);
constexpr std::size_t DEFAULT_SIZES_BUFFER_SIZE            = 1024 * 1024 * 3;
constexpr std::size_t DEFAULT_HEADER_BUFFER_SIZE           = 1024 * 1024 * 10;
constexpr std::size_t DEFAULT_PAYLOAD_BUFFER_SIZE          = 1024 * 1024 * 1024;

/**
 * @brief Transform DOCA GPUNetIO raw packets into Dataframe for other Morpheus stages.
 *
 * Tested only on ConnectX 6-Dx with a single GPU on the same NUMA node running firmware 24.35.2000
 */
class MORPHEUS_EXPORT DocaConvertStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<RawPacketMessage>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<RawPacketMessage>, std::shared_ptr<MessageMeta>>;
    // Input = Receive = Sink = RawPacketMessage
    using typename base_t::sink_type_t;
    // Output = Send = Source = MessageMeta
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    DocaConvertStage(std::chrono::milliseconds max_time_delta = DEFAULT_MAX_TIME_DELTA,
                     std::size_t sizes_buffer_size            = DEFAULT_SIZES_BUFFER_SIZE,
                     std::size_t header_buffer_size           = DEFAULT_HEADER_BUFFER_SIZE,
                     std::size_t payload_buffer_size          = DEFAULT_PAYLOAD_BUFFER_SIZE);
    ~DocaConvertStage() override;

  private:
    subscribe_fn_t build();

    /**
     * Called every time a message is passed to this stage
     */
    void on_raw_packet_message(sink_type_t x);
    void send_buffered_data(rxcpp::subscriber<source_type_t>& output, doca::PacketDataBuffer&& paccket_buffer);
    void buffer_reader(rxcpp::subscriber<source_type_t>& output);

    cudaStream_t m_stream;
    rmm::cuda_stream_view m_stream_cpp;

    std::chrono::milliseconds m_max_time_delta;
    std::size_t m_payload_buffer_size;
    std::shared_ptr<mrc::BufferedChannel<doca::PacketDataBuffer>> m_buffer_channel;
};

/****** DocaConvertStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT DocaConvertStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DocaConvertStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<DocaConvertStage>> init(
        mrc::segment::Builder& builder,
        std::string const& name,
        std::chrono::milliseconds max_time_delta = DEFAULT_MAX_TIME_DELTA,
        std::size_t sizes_buffer_size            = DEFAULT_SIZES_BUFFER_SIZE,
        std::size_t header_buffer_size           = DEFAULT_HEADER_BUFFER_SIZE,
        std::size_t payload_buffer_size          = DEFAULT_PAYLOAD_BUFFER_SIZE);
};

}  // namespace morpheus
