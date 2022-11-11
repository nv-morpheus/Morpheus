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

/*
 *                                                                                              ┌───┐ ┌────┐ ┌────┐
 *                                                                         MATCHED TRAFFIC      │DPI│ │    │ │    │
 *       APP_INIT_&_OFFLOAD_RULES_DIAGRAM                                  ┌────────────────────┤WORKERS   │ │    │
 *                                                                         │ SET STATE TO       │   │ │    │ │    │
 *                                                                         │ HAIRPIN/DROP       │   │ │    │ │    │
 *                                                                         │                    │   │ │    │ │    │
 *     ┌───────────────────────────────────────────────────────────────────┼────────────────────┼───┼─┼────┼─┼────┼──┐
 *     │                                                                   │                    │   │ │    │ │    │  │
 *     │                                                                   │                    │   │ │    │ │    │  │
 *     │     NIC HW STEERING                                               │                    └─▲─┘ └──▲─┘ └──▲─┘  │
 *     │                                                                   │                      │      │      │    │
 *     │                                                                   │                      │      │      │    │
 *     │                                                                   │                      │      │      │    │
 *     │                                                                   │                      │      │      │    │
 *     │                                                                   ▼            ┌─────────┴──────┤      │    │
 *     │                     RTE_FLOW                RTE_SFT            RTE_SFT         │                ├──────┘    │
 *     │                                                                                │      RSS       │           │
 *     │                 ┌──────────────┐         ┌────────────┐      ┌──────────┐      │                │           │
 *     │                 │              │         │            │      │ POST_SFT │      └────────────────┘           │
 *     │                 │  SFT ACTION  │         │ MARK STATE │      │          │              ▲                    │
 *     │                 │  JUMP TO     │         │ IN SFT     │      │ CHECK    │              │                    │
 *     │    L4 TRAFFIC   │  TABLE WITH  ├────────►│            ├─────►│ VALID FID├──────────────┘                    │
 *     │ ┌─────────────► │  PREDEFINED  │         │            │      │ &&       │                                   │
 *     │ │               │  ZONE        │         │            │      │ VALID    │                                   │
 *     │ │               │              │         │            │      │   STATE  │                                   │
 *     │ │               └──────────────┘         └────────────┘      └┬─────────┘                                   │
 *     │ │                                                             │                                             │
 *     │ │                                                             │                                             │
 *     │ │                                                             │HAIRPIN MATCHED                              │
 *┌────┼─┴┐                                                            │  TRAFFIC      ┌─────────┐                   │
 *│    │  │                                                            └───────────────►         │              ┌────┼──┐
 *│ PORT  │                         NON L4 TRAFFIC                                     │  HAIRPIN│              │    │  │
 *│ RX │  ├────────────────────────────────────────────────────────────────────────────►         │              │  PORT │
 *│    │  │                                                                            │  QUEUE  ├─────────────►│  TX│  │
 *└────┼──┘                                                                            │         │              │    │  │
 *     │                                                                               └─────────┘              └────┼──┘
 *     │_____________________________________________________________________________________________________________│
 *
 */

#ifndef COMMON_OFFLOAD_RULES_H_
#define COMMON_OFFLOAD_RULES_H_

#include <rte_flow.h>

#include <doca_error.h>

#ifdef GPU_SUPPORT
#include "gpu_init.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SFT_PORTS_NUM (2)		/* Number of ports that used for SFT offload rules implementation */
#define SFT_IP_VERSIONS_NUM (2)		/* Number of IP protocol versions {IPV4, IPV6} */
#define SFT_IP_PROTOCOLS_NUM (2)	/* Number of IP protocols {UDP, TCP} */
#define SFT_POST_RULES_NUM (2)		/* Number of post hairpin rules {Matched Flow , Drop Flow} */

/* SFT rules priorities */
enum POST_SFT_GROUP_PRIORITY
{
	SET_STATE_PRIORITY,	/* Priority for rules with SFT state */
	SFT_TO_RSS_PRIORITY,	/* Priority for rules with no SFT state */
};

/* Pre SFT rules priorities */
enum PRE_SFT_GROUP_PRIORITY
{
	JUMP_TO_SFT_PRIORITY = 0,	/* Priority for rules that should be handled by SFT */
	HAIRPIN_NON_L4_PRIORITY = 3,	/* Priority for rules that should skip SFT */
};

/* SFT states */
enum SFT_USER_STATE
{
	RSS_FLOW = 0,			/* Flow should be forwarded to RSS */
	HAIRPIN_MATCHED_FLOW = 1,	/* Flow should be hairpinned */
	/* Flow should be hairpinned,
	 * skipped by the DPI worker after predefined number of bytes have been scanned
	 * See max_dpi_depth in dpi_worker.h
	 */
	HAIRPIN_SKIPPED_FLOW = 1,
	DROP_FLOW = 2,			/* Flow should be dropped */
};

/* Port configuration */
struct application_port_config {
	int nb_ports;			/* Set on init to 0 for don't care, required ports otherwise */
	int nb_queues;			/* Set on init to 0 for don't care, required minimum cores otherwise */
	int nb_hairpin_q;		/* Set on init to 0 to disable, hairpin queues otherwise */
	uint16_t rss_support	:1;	/* Set on init to 0 for no RSS support, RSS support otherwise */
	uint16_t lpbk_support	:1;	/* Enable loopback support */
	uint16_t isolated_mode	:1;	/* Set on init to 0 for no isolation, isolated mode otherwise */
};

/* SFT configuration */
struct application_sft_config {
	bool enable;	  		/* Enable SFT */
	bool enable_ct;	  		/* Enable connection tracking feature of SFT */
	bool enable_frag; 		/* Enable fragmentation feature of SFT  */
	bool enable_state_hairpin;	/* Enable HW hairpin offload */
	bool enable_state_drop;		/* Enable HW drop offload */
};

/* DPDK configuration */
struct application_dpdk_config {
	struct application_port_config port_config;	/* DPDK port configuration */
	struct application_sft_config sft_config;	/* DPDK SFT configuration */
	bool reserve_main_thread;			/* Reserve lcore for the main thread */
#ifdef GPU_SUPPORT
	struct gpu_pipeline pipe;			/* GPU pipeline */
#endif
};

/* rte_flow rules */
struct application_rules {
	/* Rules for pre SFT */
	struct rte_flow *set_jump_to_sft_action[SFT_PORTS_NUM * SFT_IP_PROTOCOLS_NUM * SFT_IP_VERSIONS_NUM];
	/* Rules for hairpin SFT traffic */
	struct rte_flow *query_hairpin[SFT_PORTS_NUM * SFT_POST_RULES_NUM];
	/* Rules for packets with no SFT state */
	struct rte_flow *rss_non_state[SFT_PORTS_NUM];
	/* Rules for non L4 traffic, offloaded by default */
	struct rte_flow *hairpin_non_l4[SFT_PORTS_NUM];
	/* Rules to forward fragmented packets when fragmentation is not enabled */
	struct rte_flow *forward_fragments[SFT_PORTS_NUM * SFT_IP_VERSIONS_NUM];
};

/*
 * Create application rules according to configuration
 *
 * @app_dpdk_config [in]: application DPDK configuration values
 * @action_rss [in]: RSS configuration for non-hairpin traffic
 * @action_rss_hairpin [in]: RSS Hairpin configuration
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t create_rules_sft_offload(const struct application_dpdk_config *app_dpdk_config,
				struct rte_flow_action_rss *action_rss, struct rte_flow_action_rss *action_rss_hairpin);

/*
 * Print offload rules statistics
 */
void print_offload_rules_counter();

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMMON_OFFLOAD_RULES_H_ */
