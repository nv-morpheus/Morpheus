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

#include "morpheus/doca/packet_data_buffer.hpp"
#include "morpheus/export.h"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <cuda_runtime.h>  // for cudaStream_t
#include <mrc/channel/buffered_channel.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>  // for Object
#include <pymrc/node.hpp>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_view
#include <rxcpp/rx.hpp>              // for subscriber

#include <chrono>
#include <cstddef>  // for size_t
#include <memory>
#include <string>

namespace morpheus {

constexpr std::chrono::milliseconds DefaultMaxTimeDelta = std::chrono::seconds(3);
constexpr std::size_t DefaultBufferChannelSize          = 1024;

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

    DocaConvertStage(std::chrono::milliseconds max_time_delta = DefaultMaxTimeDelta,
                     std::size_t buffer_channel_size          = DefaultBufferChannelSize);
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
        std::chrono::milliseconds max_time_delta = DefaultMaxTimeDelta,
        std::size_t buffer_channel_size          = DefaultBufferChannelSize);
};

}  // namespace morpheus
