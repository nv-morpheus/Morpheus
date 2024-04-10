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
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/raw_packet.hpp"

#include <mrc/segment/builder.hpp>
#include <pymrc/node.hpp>

#include <memory>

namespace morpheus {

namespace doca {

struct DocaContext;
struct DocaRxQueue;
struct DocaRxPipe;
struct DocaSemaphore;
}  // namespace doca

#pragma GCC visibility push(default)

/**
 * @brief Receives a firehose of raw packets from a GPUNetIO-enabled device.
 *
 * Tested only on ConnectX 6-Dx with a single GPU on the same NUMA node running firmware 24.35.2000
 */
class DocaSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<RawPacketMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<RawPacketMessage>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    DocaSourceStage(std::string const& nic_pci_address,
                    std::string const& gpu_pci_address,
                    std::string const& traffic_type);
    virtual ~DocaSourceStage();

  private:
    subscriber_fn_t build();

    std::shared_ptr<morpheus::doca::DocaContext> m_context;
    std::vector<std::shared_ptr<morpheus::doca::DocaRxQueue>> m_rxq;
    std::vector<std::shared_ptr<morpheus::doca::DocaSemaphore>> m_semaphore;
    std::shared_ptr<morpheus::doca::DocaRxPipe> m_rxpipe;
    enum doca_traffic_type m_traffic_type;
};

/****** DocaSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DocaSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DocaSourceStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<DocaSourceStage>> init(mrc::segment::Builder& builder,
                                                                       std::string const& name,
                                                                       std::string const& nic_pci_address,
                                                                       std::string const& gpu_pci_address,
                                                                       std::string const& traffic_type);
};

/**
 * @brief Transform DOCA GPUNetIO raw packets into Dataframe for other Morpheus stages.
 *
 * Tested only on ConnectX 6-Dx with a single GPU on the same NUMA node running firmware 24.35.2000
 */
class DocaConvertStage : public mrc::pymrc::PythonNode<std::shared_ptr<RawPacketMessage>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<RawPacketMessage>, std::shared_ptr<MessageMeta>>;
    // Input = Receive = Sink = RawPacketMessage
    using typename base_t::sink_type_t;
    // Output = Send = Source = MessageMeta
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    DocaConvertStage(bool const& split_hdr);
    virtual ~DocaConvertStage();

  private:
    // subscribe_fn_t build();
    /**
     * Called every time a message is passed to this stage
     */
    source_type_t on_data(sink_type_t x);
    source_type_t on_raw_packet_message(sink_type_t x);

    bool m_split_hdr;
    cudaStream_t m_stream;
    rmm::cuda_stream_view m_stream_cpp;
};

/****** DocaConvertStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct DocaConvertStageInterfaceProxy
{
    /**
     * @brief Create and initialize a DocaConvertStage, and return the result.
     */
    static std::shared_ptr<mrc::segment::Object<DocaConvertStage>> init(mrc::segment::Builder& builder,
                                                                        std::string const& name,
                                                                        const bool& split_hdr);
};

#pragma GCC visibility pop

}  // namespace morpheus
