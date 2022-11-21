/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include "morpheus/doca/common.h"
// #include <

DOCA_LOG_REGISTER(GPU_FLOWS);

#define NB_ACTION_ARRAY (1)

struct doca_flow_port *
init_doca_flow(uint8_t port_id, uint8_t rxq_num, struct application_dpdk_config *dpdk_config)
{
	char port_id_str[MAX_PORT_STR_LEN];
	struct doca_flow_error err = {0};
	struct doca_flow_port_cfg port_cfg = {0};
	struct doca_flow_port *df_port;
	struct doca_flow_cfg rxq_flow_cfg = {0};

	if (dpdk_config == NULL)
		return NULL;

	/*
	 * DPDK should be initialized before DOCA Flow.
	 * Default RSS is not applied to external DPDK RxQs
	 */
	// dpdk_queues_and_ports_init(dpdk_config);

    struct rte_eth_dev_info dev_info = {0};
    rte_eth_dev_info_get(port_id, &dev_info);
    struct rte_eth_conf eth_conf = {0};
    auto ret_0 = rte_eth_dev_configure(port_id, 1, 0, &eth_conf);
    rte_eth_dev_set_mtu(port_id, 8192);

    struct rte_mempool* mp = rte_pktmbuf_pool_create("TEST", 8192, 0, 0, 2048, rte_eth_dev_socket_id(port_id));
    auto ret_1 = rte_eth_rx_queue_setup(port_id, 0, 2048, rte_eth_dev_socket_id(port_id), NULL, mp);

    rte_eth_dev_start(port_id);


	/* Initialize doca flow framework */
	rxq_flow_cfg.queues = rxq_num;
	rxq_flow_cfg.mode_args = "vnf";
	if (doca_flow_init(&rxq_flow_cfg, &err) < 0) {
		DOCA_LOG_ERR("failed to init doca flow: %s", err.message);
		return NULL;
	}

	/* Start doca flow port */
	port_cfg.port_id = port_id;
	port_cfg.type = DOCA_FLOW_PORT_DPDK_BY_ID;
	snprintf(port_id_str, MAX_PORT_STR_LEN, "%d", port_cfg.port_id);
	port_cfg.devargs = port_id_str;
	df_port = doca_flow_port_start(&port_cfg, &err);
	if (df_port == NULL)
		DOCA_LOG_ERR("failed to initialize doca flow port: %s", err.message);

	return df_port;
}

/* Builds DOCA flow pipe for every RxQ with an incremental src IP address */
struct doca_flow_pipe *
build_rxq_pipe(
    uint16_t port_id,
    struct doca_flow_port *port,
    uint32_t source_ip_filter,
    uint16_t dpdk_rxq_idx)
{
	struct doca_flow_match rxq_match;
	struct doca_flow_fwd rxq_fw;
	struct doca_flow_actions actions;
    struct doca_flow_actions *actions_array[NB_ACTION_ARRAY];
	struct doca_flow_pipe_cfg rxq_pipe_cfg;
	struct doca_flow_pipe *rxq_pipe;
	struct doca_flow_error err = {0};
	uint16_t rss_queues[1];
	struct doca_flow_pipe_entry *entry;

	if (port == NULL)
		return NULL;

	memset(&actions, 0, sizeof(actions));
	memset(&rxq_fw, 0, sizeof(rxq_fw));
	memset(&rxq_match, 0, sizeof(rxq_match));
	memset(&rxq_pipe_cfg, 0, sizeof(rxq_pipe_cfg));

	rxq_pipe_cfg.attr.name = "GPU_RXQ_FW_PIPE";
	rxq_pipe_cfg.attr.type = DOCA_FLOW_PIPE_BASIC;
	rxq_pipe_cfg.attr.nb_actions = 1;
	rxq_pipe_cfg.attr.is_root = true;
	actions_array[0] = &actions;
	rxq_pipe_cfg.actions = actions_array;
	rxq_pipe_cfg.match = &rxq_match;
	rxq_pipe_cfg.port = port;

	rxq_match.out_src_ip.type = DOCA_FLOW_IP4_ADDR;
	rxq_match.out_src_ip.ipv4_addr = source_ip_filter;
	rxq_match.out_l4_type = IPPROTO_TCP;

	rss_queues[0] = dpdk_rxq_idx;
	rxq_fw.type = DOCA_FLOW_FWD_RSS;
	rxq_fw.rss_queues = rss_queues;
	rxq_fw.rss_flags = DOCA_FLOW_RSS_IP;
	rxq_fw.num_of_queues = 1;

	rxq_pipe = doca_flow_pipe_create(&rxq_pipe_cfg, &rxq_fw, NULL, &err);
	if (rxq_pipe == NULL) {
		DOCA_LOG_ERR("RxQ pipe creation FAILED: %s", err.message);
		return NULL;
	}

	/* Add HW offload */
	entry = doca_flow_pipe_add_entry(0, rxq_pipe, &rxq_match, &actions, NULL, NULL, 0, NULL, &err);
	if (entry == NULL) {
		DOCA_LOG_ERR("RxQ pipe entry creation FAILED: %s", err.message);
		// doca_flow_destroy_pipe(port_id, rxq_pipe);
		return NULL;
	}

	return rxq_pipe;
}

doca_error_t
destroy_doca_flow(uint8_t port_id, uint8_t rxq_num, struct doca_flow_pipe **rxq_pipe)
{
	if (rxq_pipe == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	// for (int rxq_idx = 0; rxq_idx < rxq_num; rxq_idx++)
	// 	doca_flow_destroy_pipe(port_id, rxq_pipe[rxq_idx]);

	doca_flow_destroy();

	return DOCA_SUCCESS;

}
