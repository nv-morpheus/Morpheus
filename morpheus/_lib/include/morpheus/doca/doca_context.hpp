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

#define DOCA_ALLOW_EXPERIMENTAL_API

#include <doca_eth_rxq.h>
#include <doca_flow.h>
#include <doca_gpunetio.h>
#include <morpheus/doca/error.hpp>
#include <morpheus/doca/rte_context.hpp>

#include <memory>
#include <string>
#include <type_traits>

#define GPU_PAGE_SIZE (1UL << 16)

namespace morpheus::doca {

#pragma GCC visibility push(default)

struct DocaContext
{
  private:
    doca_gpu* m_gpu;
    doca_dev* m_dev;
    doca_pci_bdf m_pci_bdf;
    doca_flow_port* m_flow_port;
    uint16_t m_nic_port;
    uint32_t m_max_queue_count;
    std::unique_ptr<RTEContext> m_rte_context;

  public:
    DocaContext(std::string nic_addr, std::string gpu_addr);
    ~DocaContext();

    doca_gpu* gpu();
    doca_dev* dev();
    uint16_t nic_port();
    doca_flow_port* flow_port();
};

#pragma GCC visibility pop

}  // namespace morpheus::doca
