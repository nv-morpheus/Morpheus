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

#include "doca_rx_queue.hpp"

#include "common.hpp"
#include "error.hpp"

#include "morpheus/utilities/error.hpp"

#include <glog/logging.h>

namespace morpheus::doca {

DocaRxQueue::DocaRxQueue(std::shared_ptr<DocaContext> context) :
  m_context(context),
  m_rxq_info_gpu(nullptr),
  m_rxq_info_cpu(nullptr),
  m_packet_buffer(nullptr),
  m_doca_ctx(nullptr)
{
    DOCA_TRY(doca_eth_rxq_create(&m_rxq_info_cpu));
    DOCA_TRY(doca_eth_rxq_set_num_packets(m_rxq_info_cpu, MAX_PKT_NUM));
    DOCA_TRY(doca_eth_rxq_set_max_packet_size(m_rxq_info_cpu, MAX_PKT_SIZE));
    uint32_t cyclic_buffer_size;
    DOCA_TRY(doca_eth_rxq_get_pkt_buffer_size(m_rxq_info_cpu, &cyclic_buffer_size));
    DOCA_TRY(doca_mmap_create(nullptr, &m_packet_buffer));
    DOCA_TRY(doca_mmap_dev_add(m_packet_buffer, context->dev()));

    m_packet_mem = std::make_unique<DocaMem<void>>(m_context, cyclic_buffer_size, DOCA_GPU_MEM_GPU);

    DOCA_TRY(doca_mmap_set_memrange(m_packet_buffer, m_packet_mem->gpu_ptr(), cyclic_buffer_size));
    DOCA_TRY(doca_mmap_set_permissions(m_packet_buffer, DOCA_ACCESS_LOCAL_READ_WRITE));
    DOCA_TRY(doca_mmap_start(m_packet_buffer));
    DOCA_TRY(doca_eth_rxq_set_pkt_buffer(m_rxq_info_cpu, m_packet_buffer, 0, cyclic_buffer_size));

    m_doca_ctx = doca_eth_rxq_as_doca_ctx(m_rxq_info_cpu);

    if (m_doca_ctx == nullptr)
    {
        MORPHEUS_FAIL("unable to rxq as doca ctx ?");
    }

    DOCA_TRY(doca_ctx_dev_add(m_doca_ctx, context->dev()));
    DOCA_TRY(doca_ctx_set_datapath_on_gpu(m_doca_ctx, context->gpu()));
    DOCA_TRY(doca_ctx_start(m_doca_ctx));
    DOCA_TRY(doca_eth_rxq_get_gpu_handle(m_rxq_info_cpu, &m_rxq_info_gpu));
}

DocaRxQueue::~DocaRxQueue()
{
    if (m_doca_ctx != nullptr)
    {
        auto doca_ret = doca_ctx_dev_rm(m_doca_ctx, m_context->dev());
        if (doca_ret != DOCA_SUCCESS)
        {
            LOG(WARNING) << "doca_ctx_dev_rm failed (" << doca_ret << ")" << std::endl;
        }
    }

    if (m_rxq_info_cpu != nullptr)
    {
        auto doca_ret = doca_eth_rxq_destroy(m_rxq_info_cpu);
        if (doca_ret != DOCA_SUCCESS)
        {
            LOG(WARNING) << "doca_eth_rxq_destroy failed (" << doca_ret << ")" << std::endl;
        }
    }

    if (m_packet_buffer != nullptr)
    {
        auto doca_ret = doca_mmap_destroy(m_packet_buffer);
        if (doca_ret != DOCA_SUCCESS)
        {
            LOG(WARNING) << "doca_mmap_destroy failed (" << doca_ret << ")" << std::endl;
        }
    }
}

doca_eth_rxq* DocaRxQueue::rxq_info_cpu()
{
    return m_rxq_info_cpu;
}

doca_gpu_eth_rxq* DocaRxQueue::rxq_info_gpu()
{
    return m_rxq_info_gpu;
}

}  // namespace morpheus::doca
