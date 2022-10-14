/*
 * Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_sft.h>

#include <doca_log.h>

#include "morpheus/doca/dpdk_utils.h"
#include "morpheus/doca/utils.h"

#ifdef GPU_SUPPORT
#include "morpheus/doca/gpu_init.h"
#endif

DOCA_LOG_REGISTER(NUTILS);

#define RSS_KEY_LEN 40

#ifndef IPv6_BYTES
#define IPv6_BYTES_FMT "%02x%02x:%02x%02x:%02x%02x:%02x%02x:"\
			"%02x%02x:%02x%02x:%02x%02x:%02x%02x"
#define IPv6_BYTES(addr) \
	addr[0],  addr[1], addr[2],  addr[3], addr[4],  addr[5], addr[6],  addr[7], \
	addr[8],  addr[9], addr[10], addr[11], addr[12], addr[13], addr[14], addr[15]
#endif


static int
bind_hairpin_queues(uint16_t port_id)
{
	/* Configure the Rx and Tx hairpin queues for the selected port. */
	int ret = 0, peer_port, peer_ports_len;
	uint16_t peer_ports[RTE_MAX_ETHPORTS];

	/* bind current Tx to all peer Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 1);
	if (peer_ports_len < 0)
		return peer_ports_len;
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		ret = rte_eth_hairpin_bind(port_id, peer_ports[peer_port]);
		if (ret < 0)
			return ret;
	}
	/* bind all peer Tx to current Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 0);
	if (peer_ports_len < 0)
		return peer_ports_len;
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		ret = rte_eth_hairpin_bind(peer_ports[peer_port], port_id);
		if (ret < 0)
			return ret;
	}
	return ret;
}

static int
unbind_hairpin_queues(uint16_t port_id)
{
	/* Configure the Rx and Tx hairpin queues for the selected port. */
	int ret = 0, peer_port, peer_ports_len;
	uint16_t peer_ports[RTE_MAX_ETHPORTS];

	/* unbind current Tx from all peer Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 1);
	if (peer_ports_len < 0)
		return peer_ports_len;
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		ret = rte_eth_hairpin_unbind(port_id, peer_ports[peer_port]);
		if (ret < 0)
			return ret;
	}
	/* unbind all peer Tx from current Rx */
	peer_ports_len = rte_eth_hairpin_get_peer_ports(port_id, peer_ports, RTE_MAX_ETHPORTS, 0);
	if (peer_ports_len < 0)
		return peer_ports_len;
	for (peer_port = 0; peer_port < peer_ports_len; peer_port++) {
		ret = rte_eth_hairpin_unbind(peer_ports[peer_port], port_id);
		if (ret < 0)
			return ret;
	}
	return ret;
}

static int
setup_hairpin_queues(uint16_t port_id, uint16_t peer_port_id, uint16_t *reserved_hairpin_q_list, int hairpin_queue_len)
{
	/* Port:
	 *	0. RX queue
	 *	1. RX hairpin queue rte_eth_rx_hairpin_queue_setup
	 *	2. TX hairpin queue rte_eth_tx_hairpin_queue_setup
	 */

	int ret = 0, hairpin_q;
	uint16_t nb_tx_rx_desc = 2048;
	uint32_t manual = 1;
	uint32_t tx_exp = 1;
	struct rte_eth_hairpin_conf hairpin_conf = {
	    .peer_count = 1,
	    .manual_bind = !!manual,
	    .tx_explicit = !!tx_exp,
	    .peers[0] = {peer_port_id},
	};

	for (hairpin_q = 0; hairpin_q < hairpin_queue_len; hairpin_q++) {
		// TX
		hairpin_conf.peers[0].queue = reserved_hairpin_q_list[hairpin_q];
		ret = rte_eth_tx_hairpin_queue_setup(port_id, reserved_hairpin_q_list[hairpin_q], nb_tx_rx_desc,
						     &hairpin_conf);
		if (ret != 0)
			return ret;
		// RX
		hairpin_conf.peers[0].queue = reserved_hairpin_q_list[hairpin_q];
		ret = rte_eth_rx_hairpin_queue_setup(port_id, reserved_hairpin_q_list[hairpin_q], nb_tx_rx_desc,
						     &hairpin_conf);
		if (ret != 0)
			return ret;
	}
	return ret;
}

static void
enable_hairpin_queues(uint8_t nb_ports)
{
	uint8_t port_id;

	for (port_id = 0; port_id < nb_ports; port_id++)
		if (bind_hairpin_queues(port_id) != 0)
			APP_EXIT("Hairpin bind failed on port=%u", port_id);
}

static void
update_flow_action_rss(struct rte_flow_action_rss *action_rss, const uint16_t *queue_list, uint8_t nb_queues,
		       const struct rte_eth_rss_conf *rss_conf, enum rte_eth_hash_function hash_func, uint32_t level)
{
	action_rss->queue_num = nb_queues;
	action_rss->queue = queue_list;
	action_rss->types = rss_conf->rss_hf;
	action_rss->key_len = rss_conf->rss_key_len;
	action_rss->key = rss_conf->rss_key;
	action_rss->func = hash_func;
	action_rss->level = level;
}

static void
dpdk_sft_init(const struct application_dpdk_config *app_dpdk_config)
{
	int ret = 0;
	uint8_t port_id = 0;
	uint8_t queue_index;
	uint8_t nb_queues = app_dpdk_config->port_config.nb_queues;
	uint8_t nb_hairpin_q = app_dpdk_config->port_config.nb_hairpin_q;
	uint16_t queue_list[nb_queues];
	uint16_t hairpin_queue_list[nb_hairpin_q];
	uint32_t level = 0;
	struct rte_sft_conf sft_config = {
	    .nb_queues = nb_queues,					/* Num of queues */
	    .nb_max_entries = (1 << 20),				/* Max number of connections */
	    .tcp_ct_enable = app_dpdk_config->sft_config.enable_ct,	/* Enable TCP Connection Tracking */
	    .ipfrag_enable = app_dpdk_config->sft_config.enable_frag,	/* Enable fragmented packets */
	    .reorder_enable = 1,					/* Enable reorder */
	    .default_aging = 60,					/* Default aging */
	    .nb_max_ipfrag = 4096,					/* Max of IP fragments */
	    .app_data_len = 1,						/* Application's data length */
	};
	struct rte_flow_action_rss action_rss;
	struct rte_flow_action_rss action_rss_hairpin;
	struct rte_sft_error sft_error;
	uint8_t rss_key[RSS_KEY_LEN];
	struct rte_eth_rss_conf rss_conf = {
	    .rss_key = rss_key,
	    .rss_key_len = RSS_KEY_LEN,
	};

	ret = rte_sft_init(&sft_config, &sft_error);
	if (ret < 0)
		APP_EXIT("SFT init failed");

	ret = rte_eth_dev_rss_hash_conf_get(port_id, &rss_conf);
	if (ret != 0)
		APP_EXIT("Get port RSS configuration failed, ret=%d", ret);

	for (queue_index = 0; queue_index < nb_queues; queue_index++)
		queue_list[queue_index] = queue_index;
	if (sft_config.ipfrag_enable)
		rss_conf.rss_hf = ETH_RSS_IP;
	update_flow_action_rss(&action_rss, queue_list, nb_queues, &rss_conf, RTE_ETH_HASH_FUNCTION_DEFAULT, level);

	for (queue_index = 0; queue_index < nb_hairpin_q; queue_index++)
		hairpin_queue_list[queue_index] = nb_queues + queue_index;
	update_flow_action_rss(&action_rss_hairpin, hairpin_queue_list, nb_hairpin_q, &rss_conf,
			       RTE_ETH_HASH_FUNCTION_DEFAULT, level);

	create_rules_sft_offload(app_dpdk_config, &action_rss, &action_rss_hairpin);

}

static struct rte_mempool *
allocate_mempool(const uint32_t total_nb_mbufs)
{
	struct rte_mempool *mbuf_pool;
	/* Creates a new mempool in memory to hold the mbufs */
	mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", total_nb_mbufs, MBUF_CACHE_SIZE, 0,
					    RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	if (mbuf_pool == NULL)
		APP_EXIT("Cannot allocate mbuf pool");
	return mbuf_pool;
}

static int
port_init(struct rte_mempool *mbuf_pool, uint8_t port, struct application_dpdk_config *app_config)
{
	int ret;
	int symmetric_hash_key_length = RSS_KEY_LEN;
	const uint16_t nb_hairpin_queues = app_config->port_config.nb_hairpin_q;
	const uint16_t rx_rings = app_config->port_config.nb_queues;
	const uint16_t tx_rings = app_config->port_config.nb_queues;
	const uint16_t rss_support = !!(app_config->port_config.rss_support &&
				       (app_config->port_config.nb_queues > 1));
	bool isolated = !!app_config->port_config.isolated_mode;
	uint16_t q, queue_index;
	uint16_t rss_queue_list[nb_hairpin_queues];
	struct rte_ether_addr addr;
	struct rte_eth_dev_info dev_info;
	struct rte_flow_error error;
	uint8_t symmetric_hash_key[RSS_KEY_LEN] = {
	    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
	    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
	    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
	};
	const struct rte_eth_conf port_conf_default = {
	    .lpbk_mode = app_config->port_config.lpbk_support,
	    .rxmode = {
		    .max_rx_pkt_len = RTE_ETHER_MAX_LEN,
		},
	    .rx_adv_conf = {
		    .rss_conf = {
			    .rss_key_len = symmetric_hash_key_length,
			    .rss_key = symmetric_hash_key,
			    .rss_hf = (ETH_RSS_IP | ETH_RSS_UDP | ETH_RSS_TCP),
			},
		},
	};
	struct rte_eth_conf port_conf = port_conf_default;

	if (!rte_eth_dev_is_valid_port(port))
		APP_EXIT("Invalid port");
	ret = rte_eth_dev_info_get(port, &dev_info);
	if (ret != 0)
		APP_EXIT("Failed getting device (port %u) info, error=%s", port, strerror(-ret));
	port_conf.rxmode.mq_mode = rss_support ? ETH_MQ_RX_RSS : ETH_MQ_RX_NONE;

#ifdef GPU_SUPPORT
	if (app_config->pipe.gpu_support) {
		struct rte_pktmbuf_extmem *ext_mem = &app_config->pipe.ext_mem;

		/* Mapped the memory region to the devices (ports) */
		DOCA_LOG_DBG("GPU Support, DMA map to GPU");
		ret = rte_dev_dma_map(dev_info.device, ext_mem->buf_ptr, ext_mem->buf_iova, ext_mem->buf_len);
		if (ret)
			APP_EXIT("Could not DMA map EXT memory\n");
	}
#endif
	/* Configure the Ethernet device */
	ret = rte_eth_dev_configure(port, rx_rings + nb_hairpin_queues, tx_rings + nb_hairpin_queues, &port_conf);
	if (ret != 0)
		return ret;
	if (port_conf_default.rx_adv_conf.rss_conf.rss_hf != port_conf.rx_adv_conf.rss_conf.rss_hf) {
		DOCA_LOG_DBG("Port %u modified RSS hash function based on hardware support, requested:%#" PRIx64
			     " configured:%#" PRIx64 "",
			     port, port_conf_default.rx_adv_conf.rss_conf.rss_hf,
			     port_conf.rx_adv_conf.rss_conf.rss_hf);
	}

	/* Enable RX in promiscuous mode for the Ethernet device */
	ret = rte_eth_promiscuous_enable(port);
	if (ret != 0)
		return ret;

	/* Allocate and set up RX queues according to number of cores per Ethernet port */
	for (q = 0; q < rx_rings; q++) {
		ret = rte_eth_rx_queue_setup(port, q, RX_RING_SIZE, rte_eth_dev_socket_id(port), NULL, mbuf_pool);
		if (ret < 0)
			return ret;
	}

	/* Allocate and set up TX queues according to number of cores per Ethernet port */
	for (q = 0; q < tx_rings; q++) {
		ret = rte_eth_tx_queue_setup(port, q, TX_RING_SIZE, rte_eth_dev_socket_id(port), NULL);
		if (ret < 0)
			return ret;
	}

	/* Enabled hairpin queue before port start */
	if (nb_hairpin_queues) {
		for (queue_index = 0; queue_index < nb_hairpin_queues; queue_index++)
			rss_queue_list[queue_index] = app_config->port_config.nb_queues + queue_index;
		ret = setup_hairpin_queues(port, port ^ 1, rss_queue_list, nb_hairpin_queues);
		if (ret != 0)
			APP_EXIT("Cannot hairpin port %" PRIu8 ", ret=%d", port, ret);
	}

	/* Set isolated mode (true or false) before port start */
	ret = rte_flow_isolate(port, isolated, &error);
	if (ret) {
		DOCA_LOG_DBG("Port %u could not be set isolated mode to %s (%s)",
			     port, isolated ? "true" : "false", error.message);
		return ret;
	}
	if (isolated)
		DOCA_LOG_INFO("Ingress traffic on port %u is in isolated mode\n",
			      port);

	/* Start the Ethernet port */
	ret = rte_eth_dev_start(port);
	if (ret < 0)
		return ret;

	/* Display the port MAC address */
	rte_eth_macaddr_get(port, &addr);
	DOCA_LOG_DBG("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "",
		     (unsigned int)port, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2], addr.addr_bytes[3],
		     addr.addr_bytes[4], addr.addr_bytes[5]);

	/*
	 * Check that the port is on the same NUMA node as the polling thread
	 * for best performance.
	 */
	if (rte_eth_dev_socket_id(port) > 0 && rte_eth_dev_socket_id(port) != (int)rte_socket_id()) {
		DOCA_LOG_WARN("Port %u is on remote NUMA node to polling thread", port);
		DOCA_LOG_WARN("\tPerformance will not be optimal.");
	}
	return ret;
}

static int
dpdk_ports_init(struct application_dpdk_config *app_config)
{
	int ret;
	uint8_t port_id;
	const uint8_t nb_ports = app_config->port_config.nb_ports;
	const uint32_t total_nb_mbufs = app_config->port_config.nb_queues * nb_ports * NUM_MBUFS;
	struct rte_mempool *mbuf_pool;

	/* Initialize mbufs mempool */
#ifdef GPU_SUPPORT
	if (app_config->pipe.gpu_support)
		mbuf_pool = allocate_mempool_gpu(&app_config->pipe, total_nb_mbufs);
	else
		mbuf_pool = allocate_mempool(total_nb_mbufs);
#else
	mbuf_pool = allocate_mempool(total_nb_mbufs);
#endif
	/* Needed by SFT to mark packets */
	ret = rte_flow_dynf_metadata_register();
	if (ret < 0)
		APP_EXIT("Metadata register failed");

	for (port_id = 0; port_id < nb_ports; port_id++)
		if (port_init(mbuf_pool, port_id, app_config) != 0)
			APP_EXIT("Cannot init port %" PRIu8, port_id);
	return ret;
}

void
dpdk_queues_and_ports_init(struct application_dpdk_config *app_dpdk_config)
{
	int ret = 0;

	/* Check that DPDK enabled the required ports to send/receive on */
	ret = rte_eth_dev_count_avail();
	if (app_dpdk_config->port_config.nb_ports > 0 && ret != app_dpdk_config->port_config.nb_ports)
		APP_EXIT("Application will only function with %u ports, num_of_ports=%d",
			 app_dpdk_config->port_config.nb_ports, ret);

	/* Check for available logical cores */
	ret = rte_lcore_count();
	if (app_dpdk_config->port_config.nb_queues > 0 && ret < app_dpdk_config->port_config.nb_queues)
		APP_EXIT("At least %u cores are needed for the application to run, available_cores=%d",
			 app_dpdk_config->port_config.nb_queues, ret);
	else
		app_dpdk_config->port_config.nb_queues = ret;

	if (app_dpdk_config->reserve_main_thread)
		app_dpdk_config->port_config.nb_queues -= 1;
#ifdef GPU_SUPPORT
	/* Enable GPU device and initialization the resources */
	if (app_dpdk_config->pipe.gpu_support) {
		DOCA_LOG_DBG("Enabling GPU support");
		gpu_init(&app_dpdk_config->pipe);
	}
#endif

	if (app_dpdk_config->port_config.nb_ports > 0 && dpdk_ports_init(app_dpdk_config) != 0)
		APP_EXIT("Ports allocation failed");

	/* Enable hairpin queues */
	if (app_dpdk_config->port_config.nb_hairpin_q > 0)
		enable_hairpin_queues(app_dpdk_config->port_config.nb_ports);

	if (app_dpdk_config->sft_config.enable)
		dpdk_sft_init(app_dpdk_config);
}

void
dpdk_queues_and_ports_fini(struct application_dpdk_config *app_dpdk_config)
{
	int ret = 0;
	uint16_t port_id;
	struct rte_sft_error sft_error;
#ifdef GPU_SUPPORT
	if (app_dpdk_config->pipe.gpu_support) {
		DOCA_LOG_DBG("GPU support cleanup");
		gpu_fini(&(app_dpdk_config->pipe));
	}
#endif

	if (app_dpdk_config->sft_config.enable) {
		print_offload_rules_counter();
		ret = rte_sft_fini(&sft_error);
		if (ret < 0)
			DOCA_LOG_ERR("SFT fini failed, error=%d", ret);
	}

	for (port_id = 0; port_id < app_dpdk_config->port_config.nb_ports; port_id++) {
	#ifdef GPU_SUPPORT
		if (app_dpdk_config->pipe.gpu_support) {
			struct rte_eth_dev_info dev_info;
			struct rte_pktmbuf_extmem *ext_mem = &app_dpdk_config->pipe.ext_mem;

			ret = rte_eth_dev_info_get(port_id, &dev_info);
			if (ret != 0)
				APP_EXIT("Failed getting device (port %u) info, error=%s", port_id, strerror(-ret));

			ret = rte_dev_dma_unmap(dev_info.device, ext_mem->buf_ptr, ext_mem->buf_iova, ext_mem->buf_len);
			if (ret)
				APP_EXIT("Could not DMA unmap EXT memory");
		}
	#endif
		ret = unbind_hairpin_queues(port_id);
		if (ret != 0)
			DOCA_LOG_ERR("Disabling hairpin queues failed: err=%d, port=%u", ret, port_id);
	}
	for (port_id = 0; port_id < app_dpdk_config->port_config.nb_ports; port_id++) {
		ret = rte_eth_dev_stop(port_id);
		if (ret != 0)
			DOCA_LOG_ERR("rte_eth_dev_stop: err=%d, port=%u", ret, port_id);

		ret = rte_eth_dev_close(port_id);
		if (ret != 0)
			DOCA_LOG_ERR("rte_eth_dev_close: err=%d, port=%u", ret, port_id);
	}
}

static void
print_ether_addr(const struct rte_ether_addr *dmac, const struct rte_ether_addr *smac,
		 const uint32_t ethertype)
{
	char dmac_buf[RTE_ETHER_ADDR_FMT_SIZE];
	char smac_buf[RTE_ETHER_ADDR_FMT_SIZE];

	rte_ether_format_addr(dmac_buf, RTE_ETHER_ADDR_FMT_SIZE, dmac);
	rte_ether_format_addr(smac_buf, RTE_ETHER_ADDR_FMT_SIZE, smac);
	DOCA_LOG_DBG("DMAC=%s, SMAC=%s, ether_type=0x%04x", dmac_buf, smac_buf, ethertype);
}

static void
print_l2_header(const struct rte_mbuf *packet)
{
	struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);

	print_ether_addr(&eth_hdr->d_addr, &eth_hdr->s_addr, htonl(eth_hdr->ether_type) >> 16);
}

static void
print_ipv4_addr(const rte_be32_t dip, const rte_be32_t sip, const char *packet_type)
{
	DOCA_LOG_DBG("DIP=%d.%d.%d.%d, SIP=%d.%d.%d.%d, %s",
		(dip & 0xff000000)>>24,
		(dip & 0x00ff0000)>>16,
		(dip & 0x0000ff00)>>8,
		(dip & 0x000000ff),
		(sip & 0xff000000)>>24,
		(sip & 0x00ff0000)>>16,
		(sip & 0x0000ff00)>>8,
		(sip & 0x000000ff),
		packet_type);
}

static void
print_ipv6_addr(const uint8_t dst_addr[16], const uint8_t src_addr[16], const char *packet_type)
{
	DOCA_LOG_DBG("DIPv6="IPv6_BYTES_FMT", SIPv6="IPv6_BYTES_FMT", %s",
		IPv6_BYTES(dst_addr), IPv6_BYTES(src_addr), packet_type);
}

static void
print_l3_header(const struct rte_mbuf *packet)
{
	if (RTE_ETH_IS_IPV4_HDR(packet->packet_type)) {
		struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(packet,
			struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));

		print_ipv4_addr(htonl(ipv4_hdr->dst_addr), htonl(ipv4_hdr->src_addr),
				rte_get_ptype_l4_name(packet->packet_type));
	} else if (RTE_ETH_IS_IPV6_HDR(packet->packet_type)) {
		struct rte_ipv6_hdr *ipv6_hdr = rte_pktmbuf_mtod_offset(packet,
			struct rte_ipv6_hdr *, sizeof(struct rte_ether_hdr));

		print_ipv6_addr(ipv6_hdr->dst_addr, ipv6_hdr->src_addr,
				rte_get_ptype_l4_name(packet->packet_type));
	}
}

static void
print_l4_header(const struct rte_mbuf *packet)
{
	uint8_t *l4_hdr;
	struct rte_ipv4_hdr *ipv4_hdr;
	const struct rte_tcp_hdr *tcp_hdr;
	const struct rte_udp_hdr *udp_hdr;

	if (!RTE_ETH_IS_IPV4_HDR(packet->packet_type))
		return;

	ipv4_hdr = rte_pktmbuf_mtod_offset(packet,
		struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	l4_hdr = (typeof(l4_hdr))ipv4_hdr + rte_ipv4_hdr_len(ipv4_hdr);

	switch (ipv4_hdr->next_proto_id) {
	case IPPROTO_UDP:
		udp_hdr = (typeof(udp_hdr))l4_hdr;
		DOCA_LOG_DBG("UDP- DPORT %u, SPORT %u",
			rte_be_to_cpu_16(udp_hdr->dst_port),
			rte_be_to_cpu_16(udp_hdr->src_port));
	break;

	case IPPROTO_TCP:
		tcp_hdr = (typeof(tcp_hdr))l4_hdr;
		DOCA_LOG_DBG("TCP- DPORT %u, SPORT %u",
			rte_be_to_cpu_16(tcp_hdr->dst_port),
			rte_be_to_cpu_16(tcp_hdr->src_port));
	break;

	default:
		DOCA_LOG_DBG("Unsupported L4 protocol!");
	}
}

void
print_header_info(const struct rte_mbuf *packet, const bool l2, const bool l3, const bool l4)
{
	if (l2)
		print_l2_header(packet);
	if (l3)
		print_l3_header(packet);
	if (l4)
		print_l4_header(packet);
}

void
dpdk_init(int argc, char **argv)
{
	int ret;

	ret = rte_eal_init(argc, argv);
	if (ret < 0)
		APP_EXIT("EAL initialization failed");
}

void
dpdk_fini()
{
	int ret;

	ret = rte_eal_cleanup();
	if (ret < 0)
		APP_EXIT("rte eal cleanup failed, error=%d", ret);

	DOCA_LOG_DBG("DPDK fini is done");
}
