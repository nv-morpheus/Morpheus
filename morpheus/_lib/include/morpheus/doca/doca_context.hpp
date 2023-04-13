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

#include <doca_gpunetio.h>
#include <doca_flow.h>
#include <doca_eth_rxq.h>

#include <string>
#include <memory>

namespace morpheus::doca
{

#pragma GCC visibility push(default)

struct doca_context
{
private:
  doca_gpu* _gpu;
  doca_dev* _dev;
  doca_pci_bdf _pci_bdf;
  doca_flow_port* _flow_port;
  uint16_t _nic_port;
  uint32_t _max_queue_count;

public:
  doca_context(std::string nic_addr, std::string gpu_addr);
  ~doca_context();

  doca_gpu* gpu();
  doca_dev* dev();
  uint16_t nic_port();
  doca_flow_port* flow_port();
};

struct doca_rx_queue
{
  private:
    std::shared_ptr<doca_context> _context;
    doca_gpu_eth_rxq* _rxq_info_gpu;
    doca_eth_rxq* _rxq_info_cpu;
    doca_mmap* _packet_buffer;
    doca_ctx* _doca_ctx;

  public:
    doca_rx_queue(std::shared_ptr<doca_context> context);
    ~doca_rx_queue();

    doca_gpu_eth_rxq* rxq_info_gpu();
    doca_eth_rxq* rxq_info_cpu();
};

struct doca_rx_pipe
{
  private:
    std::shared_ptr<doca_context> _context;
    std::shared_ptr<doca_rx_queue> _rxq;
    doca_flow_pipe* _pipe;

  public:
    doca_rx_pipe(
      std::shared_ptr<doca_context> context,
      std::shared_ptr<doca_rx_queue> rxq,
      uint32_t source_ip_filter
    );
    ~doca_rx_pipe();
};

struct doca_semaphore
{
  private:
    std::shared_ptr<doca_context> _context;
    uint16_t _size;
    doca_gpu_semaphore* _semaphore;
    doca_gpu_semaphore_gpu* _semaphore_gpu;

  public:
    doca_semaphore(std::shared_ptr<doca_context> context, uint16_t size);
    ~doca_semaphore();

    doca_gpu_semaphore_gpu* in_gpu();
    uint16_t size();
};

#pragma GCC visibility pop

}
