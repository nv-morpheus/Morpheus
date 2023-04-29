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

#define DOCA_ALLOW_EXPERIMENTAL_API

#include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/error.hpp"
#include "morpheus/utilities/error.hpp"
// #include "morpheus/doca/common.h"
// #include "morpheus/doca/samples/common.h"

#include <cuda_runtime.h>
#include <doca_argp.h>
#include <doca_dpdk.h>
#include <doca_error.h>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_version.h>
#include <glog/logging.h>
#include <rte_eal.h>
#include <rte_ethdev.h>

#include <iostream>
#include <string>

// #define GPU_SUPPORT

namespace {

static uint64_t default_flow_timeout_usec;

#define MAX_PKT_SIZE 8192
#define FLOW_NB_COUNTERS 524228 /* 1024 x 512 */
#define MAX_PORT_STR_LEN 128    /* Maximal length of port name */
#define MAX_PKT_NUM 65536

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

using jobs_check_t = doca_error_t (*)(struct doca_devinfo*);

doca_error_t open_doca_device_with_pci(const struct doca_pci_bdf* value, jobs_check_t func, struct doca_dev** retval)
{
    struct doca_devinfo** dev_list;
    uint32_t nb_devs;
    struct doca_pci_bdf buf;
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

    // DOCA_LOG_WARN("Matching device not found");

    doca_devinfo_list_destroy(dev_list);

    return DOCA_ERROR_NOT_FOUND;
}

struct doca_flow_port* init_doca_flow(uint16_t port_id, uint8_t rxq_num)
{
    char port_id_str[MAX_PORT_STR_LEN];
    struct doca_flow_port_cfg port_cfg = {0};
    struct doca_flow_port* df_port;
    struct doca_flow_cfg rxq_flow_cfg = {0};
    struct rte_eth_dev_info dev_info  = {0};
    struct rte_eth_conf eth_conf      = {
             .rxmode =
            {
                     .mtu = 2048, /* Not really used, just to initialize DPDK */
            },
             .txmode =
            {
                     .offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM | RTE_ETH_TX_OFFLOAD_UDP_CKSUM | RTE_ETH_TX_OFFLOAD_TCP_CKSUM,
            },
    };
    struct rte_mempool* mp = nullptr;
    struct rte_eth_txconf tx_conf;
    struct rte_flow_error error;

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
        // DOCA_LOG_ERR("Failed rte_pktmbuf_pool_create with: %s", rte_strerror(-ret));
        throw std::exception();
        // return nullptr;
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
    snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
    port_cfg.devargs = port_id_str;
    DOCA_TRY(doca_flow_port_start(&port_cfg, &df_port));

    default_flow_timeout_usec = 0;

    return df_port;
}

}  // namespace

namespace morpheus::doca {

static doca_error_t get_dpdk_port_id_doca_dev(struct doca_dev* dev_input, uint16_t* port_id)
{
    struct doca_dev* dev_local = nullptr;
    struct doca_pci_bdf pci_addr_local;
    struct doca_pci_bdf pci_addr_input;
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

doca_context::doca_context(std::string nic_addr, std::string gpu_addr) : _max_queue_count(1)
{
    char* nic_addr_c = new char[nic_addr.size() + 1];
    char* gpu_addr_c = new char[gpu_addr.size() + 1];

    nic_addr_c[nic_addr.size()] = '\0';
    gpu_addr_c[gpu_addr.size()] = '\0';

    std::copy(nic_addr.begin(), nic_addr.end(), nic_addr_c);
    std::copy(gpu_addr.begin(), gpu_addr.end(), gpu_addr_c);

    auto argv = std::vector<char*>();

    argv.push_back("");  // program

    argv.push_back("-a");
    argv.push_back("00:00.0");

    // argv.push_back("-l");
    // argv.push_back("0,1,2,3,4");

    // argv.push_back("--log-level");
    // argv.push_back("eal,8");

    RTE_TRY(rte_eal_init(argv.size(), argv.data()));

    DOCA_TRY(parse_pci_addr(nic_addr_c, _pci_bdf));
    DOCA_TRY(open_doca_device_with_pci(&_pci_bdf, nullptr, &_dev));
    DOCA_TRY(doca_dpdk_port_probe(_dev, "dv_flow_en=2"));
    DOCA_TRY(get_dpdk_port_id_doca_dev(_dev, &_nic_port));

    if (_nic_port == RTE_MAX_ETHPORTS)
    {
        throw std::runtime_error("No DPDK port matches the DOCA device");
    }

    DOCA_TRY(doca_gpu_create(gpu_addr_c, &_gpu));

    _flow_port = init_doca_flow(_nic_port, _max_queue_count);

    if (_flow_port == nullptr)
    {
        throw std::runtime_error("DOCA Flow initialization failed");
    }

    delete[] nic_addr_c;
    delete[] gpu_addr_c;
}

doca_context::~doca_context()
{
    auto eal_ret = rte_eal_cleanup();
    if (eal_ret < 0)
    {
        LOG(WARNING) << "EAL cleanup failed (" << eal_ret << ")" << std::endl;
    }

    auto doca_ret = doca_gpu_destroy(_gpu);
    if (doca_ret != DOCA_SUCCESS)
    {
        LOG(WARNING) << "DOCA cleanup failed (" << doca_ret << ")" << std::endl;
    }
}

doca_gpu* doca_context::gpu()
{
    return _gpu;
}

doca_dev* doca_context::dev()
{
    return _dev;
}

uint16_t doca_context::nic_port()
{
    return _nic_port;
}

doca_flow_port* doca_context::flow_port()
{
    return _flow_port;
}

doca_rx_queue::doca_rx_queue(std::shared_ptr<doca_context> context) :
  _context(context),
  _rxq_info_gpu(nullptr),
  _rxq_info_cpu(nullptr),
  _packet_buffer(nullptr),
  _doca_ctx(nullptr)
{
    DOCA_TRY(doca_eth_rxq_create(&_rxq_info_cpu));
    DOCA_TRY(doca_eth_rxq_set_num_packets(_rxq_info_cpu, MAX_PKT_NUM));
    DOCA_TRY(doca_eth_rxq_set_max_packet_size(_rxq_info_cpu, MAX_PKT_SIZE));
    uint32_t cyclic_buffer_size;
    DOCA_TRY(doca_eth_rxq_get_pkt_buffer_size(_rxq_info_cpu, &cyclic_buffer_size));
    DOCA_TRY(doca_mmap_create(nullptr, &_packet_buffer));
    DOCA_TRY(doca_mmap_dev_add(_packet_buffer, context->dev()));

    DOCA_TRY(doca_gpu_mem_alloc(
        context->gpu(), cyclic_buffer_size, GPU_PAGE_SIZE, DOCA_GPU_MEM_GPU, &_packet_address, nullptr));

    DOCA_TRY(doca_mmap_set_memrange(_packet_buffer, _packet_address, cyclic_buffer_size));
    DOCA_TRY(doca_mmap_set_permissions(_packet_buffer, DOCA_ACCESS_LOCAL_READ_WRITE));
    DOCA_TRY(doca_mmap_start(_packet_buffer));
    DOCA_TRY(doca_eth_rxq_set_pkt_buffer(_rxq_info_cpu, _packet_buffer, 0, cyclic_buffer_size));

    _doca_ctx = doca_eth_rxq_as_doca_ctx(_rxq_info_cpu);

    if (_doca_ctx == nullptr)
    {
        MORPHEUS_FAIL("unable to rxq as doca ctx ?");
    }

    DOCA_TRY(doca_ctx_dev_add(_doca_ctx, context->dev()));
    DOCA_TRY(doca_ctx_set_datapath_on_gpu(_doca_ctx, context->gpu()));
    DOCA_TRY(doca_ctx_start(_doca_ctx));
    DOCA_TRY(doca_eth_rxq_get_gpu_handle(_rxq_info_cpu, &_rxq_info_gpu));
}

doca_rx_queue::~doca_rx_queue()
{
    // DOCA_TRY(doca_gpu_mem_free(_context->gpu(), _packet_address));
    // DOCA_TRY(doca_eth_rxq_destroy(_rxq_info_cpu));
}

doca_eth_rxq* doca_rx_queue::rxq_info_cpu()
{
    return _rxq_info_cpu;
}

doca_gpu_eth_rxq* doca_rx_queue::rxq_info_gpu()
{
    return _rxq_info_gpu;
}

doca_rx_pipe::doca_rx_pipe(std::shared_ptr<doca_context> context,
                           std::shared_ptr<doca_rx_queue> rxq,
                           uint32_t source_ip_filter) :
  _context(context),
  _rxq(rxq),
  _pipe(nullptr)
{
    auto rss_queues = std::array<uint16_t, 1>();
    doca_eth_rxq_get_flow_queue_id(rxq->rxq_info_cpu(), &(rss_queues[0]));

    doca_flow_match match_mask{0};
    doca_flow_match match{};
    match.outer.l3_type        = DOCA_FLOW_L3_TYPE_IP4;
    match.outer.ip4.next_proto = IPPROTO_TCP;
    match.outer.l4_type_ext    = DOCA_FLOW_L4_TYPE_EXT_TCP;
    // match.outer.ip4.src_ip = 0xffffffff;
    // match.outer.ip4.dst_ip = 0xffffffff;
    // match.outer.tcp.l4_port.src_port = 0xffff;
    // match.outer.tcp.l4_port.dst_port = 0xffff;

    doca_flow_fwd fwd{};
    fwd.type            = DOCA_FLOW_FWD_RSS;
    fwd.rss_outer_flags = DOCA_FLOW_RSS_IPV4 | DOCA_FLOW_RSS_TCP;
    fwd.rss_queues      = rss_queues.begin();
    fwd.num_of_queues   = 1;

    doca_flow_fwd miss_fwd{};
    miss_fwd.type = DOCA_FLOW_FWD_DROP;

    doca_flow_monitor monitor{};
    monitor.flags = DOCA_FLOW_MONITOR_COUNT;

    doca_flow_pipe_cfg pipe_cfg{};
    pipe_cfg.attr.name       = "GPU_RXQ_TCP_PIPE";
    pipe_cfg.attr.type       = DOCA_FLOW_PIPE_BASIC;
    pipe_cfg.attr.nb_actions = 0;
    pipe_cfg.attr.is_root    = false;
    pipe_cfg.match           = &match;
    pipe_cfg.match_mask      = &match_mask;
    pipe_cfg.monitor         = &monitor;
    pipe_cfg.port            = context->flow_port();

    DOCA_TRY(doca_flow_pipe_create(&pipe_cfg, &fwd, &miss_fwd, &_pipe));
}

doca_rx_pipe::~doca_rx_pipe()
{
    doca_flow_pipe_destroy(_pipe);
}

doca_semaphore::doca_semaphore(std::shared_ptr<doca_context> context, uint16_t size) : _context(context), _size(size)
{
    DOCA_TRY(doca_gpu_semaphore_create(_context->gpu(), &_semaphore));
    DOCA_TRY(doca_gpu_semaphore_set_memory_type(_semaphore, DOCA_GPU_MEM_CPU_GPU));
    DOCA_TRY(doca_gpu_semaphore_set_items_num(_semaphore, size));
    DOCA_TRY(doca_gpu_semaphore_start(_semaphore));
    DOCA_TRY(doca_gpu_semaphore_get_gpu_handle(_semaphore, &_semaphore_gpu));
}

doca_semaphore::~doca_semaphore()
{
    doca_gpu_semaphore_destroy(_semaphore);
}

doca_gpu_semaphore_gpu* doca_semaphore::gpu_ptr()
{
    return _semaphore_gpu;
}

uint16_t doca_semaphore::size()
{
    return _size;
}

}  // namespace morpheus::doca
