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

#include "doca_context.hpp"

#include "common.hpp"
#include "error.hpp"

#include "morpheus/utilities/error.hpp"

#include <cuda_runtime.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_mmap.h>
#include <doca_types.h>
#include <doca_version.h>
#include <glog/logging.h>
#include <rte_eal.h>
#include <rte_ethdev.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace {

#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_PORT_STR_LEN 128    /* Maximal length of port name */

doca_error_t parse_pci_addr(std::string const& addr, doca_pci_bdf& bdf)
{
    uint32_t tmpu;

    auto tmps = std::array<char, 4>();

    if (addr.length() != 7 || addr[2] != ':' || addr[5] != '.')
        return DOCA_ERROR_INVALID_VALUE;

    tmps[0] = addr[0];
    tmps[1] = addr[1];
    tmps[2] = '\0';
    tmpu    = strtoul(tmps.cbegin(), nullptr, 16);

    if ((tmpu & 0xFFFFFF00) != 0)
    {
        return DOCA_ERROR_INVALID_VALUE;
    }

    bdf.bus = tmpu;

    tmps[0] = addr[3];
    tmps[1] = addr[4];
    tmps[2] = '\0';
    tmpu    = strtoul(tmps.cbegin(), nullptr, 16);

    if ((tmpu & 0xFFFFFFE0) != 0)
    {
        return DOCA_ERROR_INVALID_VALUE;
    }

    bdf.device = tmpu;

    tmps[0] = addr[6];
    tmps[1] = '\0';
    tmpu    = strtoul(tmps.cbegin(), nullptr, 16);
    if ((tmpu & 0xFFFFFFF8) != 0)
        return DOCA_ERROR_INVALID_VALUE;
    bdf.function = tmpu;

    return DOCA_SUCCESS;
}

using jobs_check_t = doca_error_t (*)(doca_devinfo*);

doca_error_t open_doca_device_with_pci(const doca_pci_bdf* value, jobs_check_t func, doca_dev** retval)
{
    doca_devinfo** dev_list;
    uint32_t nb_devs;
    doca_pci_bdf buf;
    size_t i;

    /* Set default return value */
    *retval = nullptr;

    DOCA_TRY(doca_devinfo_list_create(&dev_list, &nb_devs));

    /* Search */
    for (i = 0; i < nb_devs; i++)
    {
        DOCA_TRY(doca_devinfo_get_pci_addr(dev_list[i], &buf));
        if (buf.raw == value->raw)
        {
            /* If any special capabilities are needed */
            if (func != nullptr && func(dev_list[i]) != DOCA_SUCCESS)
                continue;

            /* if device can be opened */
            auto res = doca_dev_open(dev_list[i], retval);
            if (res == DOCA_SUCCESS)
            {
                doca_devinfo_list_destroy(dev_list);
                return res;
            }
        }
    }

    doca_devinfo_list_destroy(dev_list);

    return DOCA_ERROR_NOT_FOUND;
}

doca_flow_port* init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
    std::array<char, MAX_PORT_STR_LEN> port_id_str;
    doca_flow_port_cfg port_cfg = {0};
    doca_flow_port* df_port;
    doca_flow_cfg rxq_flow_cfg = {0};
    rte_eth_dev_info dev_info  = {0};
    rte_eth_conf eth_conf      = {
             .rxmode =
            {
                     .mtu = 2048, /* Not really used, just to initialize DPDK */
            },
             .txmode =
            {
                     .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
            },
    };
    rte_mempool* mp = nullptr;
    rte_eth_txconf tx_conf;
    rte_flow_error error;

    /*
     * DPDK should be initialized and started before DOCA Flow.
     * DPDK doesn't start the device without, at least, one DPDK Rx queue.
     * DOCA Flow needs to specify in advance how many Rx queues will be used by the app.
     *
     * Following lines of code can be considered the minimum WAR for this issue.
     */

    RTE_TRY(rte_eth_dev_info_get(port_id, &dev_info));
    RTE_TRY(rte_eth_dev_configure(port_id, rxq_num, rxq_num, &eth_conf));

    mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, MAX_PKT_SIZE, rte_eth_dev_socket_id(port_id));

    if (mp == nullptr)
    {
        MORPHEUS_FAIL("rte_pktmbuf_pool_create failed.");
    }

    tx_conf = dev_info.default_txconf;
    tx_conf.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM;

    for (int idx = 0; idx < rxq_num; idx++)
    {
        RTE_TRY(rte_eth_rx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), nullptr, mp));
        RTE_TRY(rte_eth_tx_queue_setup(port_id, idx, 2048, rte_eth_dev_socket_id(port_id), &tx_conf));
    }

    RTE_TRY(rte_flow_isolate(port_id, 1, &error));
    RTE_TRY(rte_eth_dev_start(port_id));

    /* Initialize doca flow framework */
    rxq_flow_cfg.queues = rxq_num;
    /*
     * HWS: Hardware steering
     * Isolated: don't create RSS rule for DPDK created RX queues
     */
    rxq_flow_cfg.mode_args            = "vnf,hws,isolated";
    rxq_flow_cfg.resource.nb_counters = FLOW_NB_COUNTERS;

    DOCA_TRY(doca_flow_init(&rxq_flow_cfg));

    /* Start doca flow port */
    port_cfg.port_id = port_id;
    port_cfg.type    = DOCA_FLOW_PORT_DPDK_BY_ID;
    snprintf(port_id_str.begin(), MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str.cbegin();
    DOCA_TRY(doca_flow_port_start(&port_cfg, &df_port));

    return df_port;
}

}  // namespace

namespace morpheus::doca {

static doca_error_t get_dpdk_port_id_doca_dev(doca_dev* dev_input, uint16_t* port_id)
{
    doca_dev* dev_local = nullptr;
    doca_pci_bdf pci_addr_local;
    doca_pci_bdf pci_addr_input;
    uint16_t dpdk_port_id;

    if (dev_input == nullptr || port_id == nullptr)
        return DOCA_ERROR_INVALID_VALUE;

    *port_id = RTE_MAX_ETHPORTS;

    for (dpdk_port_id = 0; dpdk_port_id < RTE_MAX_ETHPORTS; dpdk_port_id++)
    {
        /* search for the probed devices */
        if (!rte_eth_dev_is_valid_port(dpdk_port_id))
            continue;

        DOCA_TRY(doca_dpdk_port_as_dev(dpdk_port_id, &dev_local));
        DOCA_TRY(doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_local), &pci_addr_local));
        DOCA_TRY(doca_devinfo_get_pci_addr(doca_dev_as_devinfo(dev_input), &pci_addr_input));

        if (pci_addr_local.raw == pci_addr_input.raw)
        {
            *port_id = dpdk_port_id;
            break;
        }
    }

    return DOCA_SUCCESS;
}

DocaContext::DocaContext(std::string nic_addr, std::string gpu_addr) : m_max_queue_count(1)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    m_rte_context = std::make_unique<RTEContext>();

    DOCA_TRY(parse_pci_addr(nic_addr_c, m_pci_bdf));
    DOCA_TRY(open_doca_device_with_pci(&m_pci_bdf, nullptr, &m_dev));
    DOCA_TRY(doca_dpdk_port_probe(m_dev, "dv_flow_en=2"));
    DOCA_TRY(get_dpdk_port_id_doca_dev(m_dev, &m_nic_port));

    if (m_nic_port == RTE_MAX_ETHPORTS)
    {
        throw std::runtime_error("No DPDK port matches the DOCA device");
    }

    DOCA_TRY(doca_gpu_create(gpu_addr_c, &m_gpu));

    m_flow_port = init_doca_flow(m_nic_port, m_max_queue_count);

    if (m_flow_port == nullptr)
    {
        throw std::runtime_error("DOCA Flow initialization failed");
    }

    delete[] nic_addr_c;
    delete[] gpu_addr_c;
}

DocaContext::~DocaContext()
{
    if (m_gpu != nullptr)
    {
        auto doca_ret = doca_gpu_destroy(m_gpu);
        if (doca_ret != DOCA_SUCCESS)
        {
            LOG(WARNING) << "DOCA cleanup failed (" << doca_ret << ")" << std::endl;
        }
    }
}

doca_gpu* DocaContext::gpu()
{
    return m_gpu;
}

doca_dev* DocaContext::dev()
{
    return m_dev;
}

uint16_t DocaContext::nic_port()
{
    return m_nic_port;
}

doca_flow_port* DocaContext::flow_port()
{
    return m_flow_port;
}

}  // namespace morpheus::doca
