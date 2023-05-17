/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/doca/doca_context.hpp>
#include <morpheus/doca/doca_rx_queue.hpp>

#include <memory>

namespace morpheus::doca {

/**
 * @brief Creates and manages the lifetime of a GPUNetIO Receive Pipe for a given GPUNetIO Receive Queue.
 *
 * Pipes are used to filter and/or forward packets to other Pipes or Receive Queues. A Root Pipe
 * is the primary Pipe where packets come in, and can then be forwarded to other Pipes. This is how
 * TCP/UDP as well as other types of filtering is done with GPUNetIO. Eventually packets will be
 * placed in a Receive Queue at which point they can be read using a Semaphore.
 *
 * In this implementation, a single Root Pipe is connected to a single TCP-filtering Pipe which is
 * then connected to the given Receive Queue.
 */
struct DocaRxPipe
{
  private:
    std::shared_ptr<DocaContext> m_context;
    std::shared_ptr<DocaRxQueue> m_rxq;
    doca_flow_pipe* m_pipe;
    doca_flow_pipe* m_root_pipe;

  public:
    DocaRxPipe(std::shared_ptr<DocaContext> context, std::shared_ptr<DocaRxQueue> rxq, uint32_t source_ip_filter);
    ~DocaRxPipe();
};

}  // namespace morpheus::doca
