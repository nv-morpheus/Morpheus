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

#include <rte_sft.h>

#include "morpheus/doca/offload_rules.h"
#include "morpheus/doca/utils.h"

DOCA_LOG_REGISTER(OFRLS);

#define GROUP_POST_SFT 1001
#define INITIAL_GROUP 0

struct application_rules app_rules;

static struct rte_flow *
create_rule_post_sft(uint16_t port_id, struct rte_flow_action_rss *action_rss, uint8_t sft_state,
			   struct rte_flow_error *error)
{
	int ret;
	struct rte_flow_item_sft sft_spec_and_mask = {.fid_valid = 1, .state = sft_state};
	struct rte_flow *flow = NULL;
	struct rte_flow_attr attr = {
	    .ingress = 1,
	    .priority = SET_STATE_PRIORITY,
	    .group = GROUP_POST_SFT,
	};
	struct rte_flow_action match_action = {.type = RTE_FLOW_ACTION_TYPE_RSS, .conf = action_rss};
	struct rte_flow_action drop_action = {.type = RTE_FLOW_ACTION_TYPE_DROP};
	struct rte_flow_item pattern[] = {
	    [0] = {.type = RTE_FLOW_ITEM_TYPE_SFT, .mask = &sft_spec_and_mask, .spec = &sft_spec_and_mask},
	    [1] = {.type = RTE_FLOW_ITEM_TYPE_END},
	};
	struct rte_flow_action action[] = {
	    [0] = {.type = RTE_FLOW_ACTION_TYPE_COUNT},
		/* When calling this function, sft_state can be only HAIRPIN_MATCHED_FLOW or DROP_FLOW */
		[1] = (sft_state == DROP_FLOW) ? drop_action : match_action,
		[2] = {.type = RTE_FLOW_ACTION_TYPE_END},
	};

	ret = rte_flow_validate(port_id, &attr, pattern, action, error);
	if (ret == 0)
		flow = rte_flow_create(port_id, &attr, pattern, action, error);
	return flow;
}

static struct rte_flow *
create_rule_forward_to_sft_by_pattern(uint8_t port_id, struct rte_flow_error *error, struct rte_flow_item **pattern)
{
	int ret;
	struct rte_flow_action_sft action_sft = {.zone = 0xcafe}; /* SFT zone is 0xcafe */
	struct rte_flow_action_jump action_jump = {.group = GROUP_POST_SFT};
	struct rte_flow *flow = NULL;
	struct rte_flow_attr attr = {
	    .ingress = 1,
	    .priority = JUMP_TO_SFT_PRIORITY,
	    .group = INITIAL_GROUP,
	};
	struct rte_flow_action action[] = {
	    [0] = {.type = RTE_FLOW_ACTION_TYPE_COUNT},
	    [1] = {.type = RTE_FLOW_ACTION_TYPE_SFT, .conf = &action_sft},
	    [2] = {.type = RTE_FLOW_ACTION_TYPE_JUMP, .conf = &action_jump},
	    [3] = {.type = RTE_FLOW_ACTION_TYPE_END}
	};
	struct rte_flow_item (*pattern_arr)[] = (struct rte_flow_item(*)[]) pattern;

	ret = rte_flow_validate(port_id, &attr, *pattern_arr, action, error);
	if (ret == 0)
		flow = rte_flow_create(port_id, &attr, *pattern_arr, action, error);
	return flow;
}

static struct rte_flow *
create_rule_forward_l4_to_sft(uint8_t port_id, uint8_t l3_protocol, uint8_t l4_protocol, struct rte_flow_error *error)
{
	struct rte_flow_item pattern[] = {
	    [0] = {.type = RTE_FLOW_ITEM_TYPE_ETH},
	    [1] = {.type = l3_protocol},
	    [2] = {.type = l4_protocol},
	    [3] = {.type = RTE_FLOW_ITEM_TYPE_END},
	};

	return create_rule_forward_to_sft_by_pattern(port_id, error, (struct rte_flow_item **)&pattern);
}

static struct rte_flow *
create_rule_forward_fragments_to_sft(uint8_t port_id, uint8_t l3_protocol, struct rte_flow_error *error)
{
	struct rte_flow_item_ipv6 ipv6_spec_and_mask = {.has_frag_ext = 1};
	struct rte_flow_item_ipv4 ipv4_spec = {.hdr.fragment_offset = rte_cpu_to_be_16(1)};
	struct rte_flow_item_ipv4 ipv4_mask_and_last = {.hdr.fragment_offset = rte_cpu_to_be_16(0x3fff)};
	struct rte_flow_item ipv4_opt = {.type = RTE_FLOW_ITEM_TYPE_IPV4,
					 .spec = &ipv4_spec,
					 .last = &ipv4_mask_and_last,
					 .mask = &ipv4_mask_and_last};
	struct rte_flow_item ipv6_opt = {
	    .type = RTE_FLOW_ITEM_TYPE_IPV6, .mask = &ipv6_spec_and_mask, .spec = &ipv6_spec_and_mask};
	struct rte_flow_item pattern[] = {
	    [0] = {.type = RTE_FLOW_ITEM_TYPE_ETH},
	    [1] = (l3_protocol == RTE_FLOW_ITEM_TYPE_IPV4) ? ipv4_opt : ipv6_opt,
	    [2] = {.type = RTE_FLOW_ITEM_TYPE_END},
	};

	return create_rule_forward_to_sft_by_pattern(port_id, error, (struct rte_flow_item **)&pattern);
}

static struct rte_flow *
create_rule_action_rss(uint16_t port_id, struct rte_flow_action_rss *action_rss, struct rte_flow_error *error,
		   struct rte_flow_attr *attr)
{
	int ret;
	struct rte_flow *flow = NULL;
	struct rte_flow_item pattern[] = {
	    [0] = {.type = RTE_FLOW_ITEM_TYPE_ETH},
	    [1] = {.type = RTE_FLOW_ITEM_TYPE_END},
	};
	struct rte_flow_action action[] = {
	    [0] = {.type = RTE_FLOW_ACTION_TYPE_COUNT},
	    [1] = {.type = RTE_FLOW_ACTION_TYPE_RSS, .conf = action_rss},
	    [2] = {.type = RTE_FLOW_ACTION_TYPE_END}
	};

	ret = rte_flow_validate(port_id, attr, pattern, action, error);
	if (ret == 0)
		flow = rte_flow_create(port_id, attr, pattern, action, error);
	return flow;
}

static struct rte_flow *
create_rule_hairpin_non_l4(uint16_t port_id, struct rte_flow_action_rss *action_rss, struct rte_flow_error *error)
{
	struct rte_flow_attr attr = {
	    .ingress = 1,
		.priority = HAIRPIN_NON_L4_PRIORITY,
	    .group = INITIAL_GROUP,
	};

	return create_rule_action_rss(port_id, action_rss, error, &attr);
}

static struct rte_flow *
create_rule_rss_post_sft(uint16_t port_id, struct rte_flow_action_rss *action_rss, struct rte_flow_error *error)
{
	struct rte_flow_attr attr = {
	    .ingress = 1,
	    .priority = SFT_TO_RSS_PRIORITY,
	    .group = GROUP_POST_SFT,
	};

	return create_rule_action_rss(port_id, action_rss, error, &attr);
}

void
create_rules_sft_offload(const struct application_dpdk_config *app_dpdk_config, struct rte_flow_action_rss *action_rss,
			 struct rte_flow_action_rss *action_rss_hairpin)
{

	uint8_t port_id = 0;
	uint8_t nb_ports = app_dpdk_config->port_config.nb_ports;
	struct rte_flow_error rte_error;

	if (nb_ports > SFT_PORTS_NUM)
		APP_EXIT("Invalid SFT ports number [%u] is greater than [%d]", nb_ports, SFT_PORTS_NUM);
	/*
	 * RTE_FLOW rules are created as list:
	 * 1. Forward IPv4/6 L4 traffic to SFT with predefined zone in group 0
	 * 2. Check traffic for state && valid fid for either hairpinned or dropped state in the SFT group
	 * 3. RSS all the L4 non-state traffic to the ARM cores
	 */

	for (port_id = 0; port_id < nb_ports; port_id++) {
		app_rules.set_jump_to_sft_action[port_id] =
		    create_rule_forward_l4_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV4, RTE_FLOW_ITEM_TYPE_UDP, &rte_error);
		if (app_rules.set_jump_to_sft_action[port_id] == NULL)
			APP_EXIT("Forward to SFT IPV4-UDP failed, error=%s", rte_error.message);

		app_rules.set_jump_to_sft_action[port_id + 2] =
		    create_rule_forward_l4_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV4, RTE_FLOW_ITEM_TYPE_TCP, &rte_error);
		if (app_rules.set_jump_to_sft_action[port_id + 2] == NULL)
			APP_EXIT("Forward to SFT IPV4-TCP failed, error=%s", rte_error.message);

		app_rules.set_jump_to_sft_action[port_id + 4] =
		    create_rule_forward_l4_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV6, RTE_FLOW_ITEM_TYPE_UDP, &rte_error);
		if (app_rules.set_jump_to_sft_action[port_id + 4] == NULL)
			APP_EXIT("Forward to SFT IPV6-UDP failed, error=%s", rte_error.message);

		app_rules.set_jump_to_sft_action[port_id + 6] =
		    create_rule_forward_l4_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV6, RTE_FLOW_ITEM_TYPE_TCP, &rte_error);
		if (app_rules.set_jump_to_sft_action[port_id + 6] == NULL)
			APP_EXIT("Forward to SFT IPV6-TCP failed, error=%s", rte_error.message);

		app_rules.rss_non_state[port_id] = create_rule_rss_post_sft(port_id, action_rss, &rte_error);
		if (app_rules.rss_non_state[port_id] == NULL)
			APP_EXIT("SFT set non state RSS failed, error=%s", rte_error.message);

		app_rules.hairpin_non_l4[port_id] =
			create_rule_hairpin_non_l4(port_id, action_rss_hairpin, &rte_error);
		if (app_rules.hairpin_non_l4[port_id] == NULL)
			APP_EXIT("Hairpin flow creation failed: %s", rte_error.message);

		if (app_dpdk_config->sft_config.enable_state_hairpin) {
			app_rules.query_hairpin[port_id] =
			    create_rule_post_sft(port_id, action_rss_hairpin, HAIRPIN_MATCHED_FLOW, &rte_error);
			if (app_rules.query_hairpin[port_id] == NULL)
				APP_EXIT("Forward fid with state, error=%s", rte_error.message);
		}

		if (app_dpdk_config->sft_config.enable_state_drop) {
			app_rules.query_hairpin[port_id + 2] =
			    create_rule_post_sft(port_id, action_rss_hairpin, DROP_FLOW, &rte_error);
			if (app_rules.query_hairpin[port_id + 2] == NULL)
				APP_EXIT("Drop fid with state, error=%s", rte_error.message);
		}

		if (app_dpdk_config->sft_config.enable_frag) {
			app_rules.forward_fragments[port_id] =
			    create_rule_forward_fragments_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV4, &rte_error);
			if (app_rules.forward_fragments[port_id] == NULL)
				APP_EXIT("Fragments to SFT IPV4 failed, error=%s", rte_error.message);

			app_rules.forward_fragments[port_id + 2] =
			    create_rule_forward_fragments_to_sft(port_id, RTE_FLOW_ITEM_TYPE_IPV6, &rte_error);
			if (app_rules.forward_fragments[port_id + 2] == NULL)
				APP_EXIT("Fragments to SFT IPV6 failed, error=%s", rte_error.message);
		}
	}
}

void
print_offload_rules_counter()
{
	uint8_t flow_i;
	uint8_t port_id;
	struct rte_flow_action action[] = {
	    [0] = {.type = RTE_FLOW_ACTION_TYPE_COUNT},
	    [1] = {.type = RTE_FLOW_ACTION_TYPE_END},
	};
	struct rte_flow_query_count count = {0};
	struct rte_flow_error rte_error;
	uint64_t total_ingress = 0;
	uint64_t total_rss = 0;
	uint64_t total_dropped = 0;
	uint64_t total_egress = 0;
	uint64_t total_ingress_non_l4 = 0;
	uint64_t total_fragments = 0;

	DOCA_LOG_DBG("------------ L4 Jump to SFT----------------");
	for (flow_i = 0; flow_i < 4; flow_i++) {
		port_id = flow_i % 2; /* Even flow record = port 0,  Odd flow record = port 1 */
		if (rte_flow_query(port_id, app_rules.set_jump_to_sft_action[flow_i], &action[0], &count, &rte_error) !=
		    0) {
			DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
		} else {
			if (flow_i < 2)
				DOCA_LOG_DBG("Port %d UDP - %lu", port_id, count.hits);
			else
				DOCA_LOG_DBG("Port %d TCP - %lu", port_id, count.hits);
			total_ingress += count.hits;
		}
	}

	DOCA_LOG_DBG("------------ IPV6 L4 Jump to SFT----------------");
	for (flow_i = 4; flow_i < 8; flow_i++) {
		port_id = flow_i % 2; /* Even flow record = port 0,  Odd flow record = port 1 */
		if (rte_flow_query(port_id, app_rules.set_jump_to_sft_action[flow_i], &action[0], &count, &rte_error) !=
		    0)
			DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
		else
			DOCA_LOG_DBG("Port %d TCP - %lu", port_id, count.hits);
		total_ingress += count.hits;

	}

	DOCA_LOG_DBG("----------Hairpin non L4 traffic-----------");
	for (flow_i = 0; flow_i < 2; flow_i++) {
		port_id = flow_i;
		if (rte_flow_query(port_id, app_rules.hairpin_non_l4[flow_i], &action[0], &count, &rte_error) != 0) {
			DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
		} else {
			DOCA_LOG_DBG("Port %d non L4- %lu", port_id, count.hits);
			total_ingress += count.hits;
			total_ingress_non_l4 += count.hits;
		}
	}

	DOCA_LOG_DBG("---------Hairpin using state post SFT -----");
	for (flow_i = 0; flow_i < 4; flow_i++) {
		port_id = flow_i % 2; /* Even flow record = port 0,  Odd flow record = port 1 */
		if (rte_flow_query(port_id, app_rules.query_hairpin[flow_i], &action[0], &count, &rte_error) != 0) {
			DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
		} else {
			if (flow_i < 2) {
				DOCA_LOG_DBG("Port %d state hairpin - %lu", port_id, count.hits);
				total_egress += count.hits;
			} else {
				DOCA_LOG_DBG("Port %d state drop - %lu", port_id, count.hits);
				total_dropped += count.hits;
			}
		}
	}

	DOCA_LOG_DBG("---------------RSS post SFT----------------");
	for (flow_i = 0; flow_i < 2; flow_i++) {
		port_id = flow_i;
		if (rte_flow_query(port_id, app_rules.rss_non_state[flow_i], &action[0], &count, &rte_error) != 0) {
			DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
		} else {
			DOCA_LOG_DBG("Port %d RSS to queues - %lu", port_id, count.hits);
			total_rss += count.hits;
		}
	}
	if (app_rules.forward_fragments[0] != NULL) {
		DOCA_LOG_DBG("---------Fragments to SFT -----");
		for (flow_i = 0; flow_i < 4; flow_i++) {
			port_id = flow_i % 2; /* Even flow record = port 0,  Odd flow record = port 1 */
			if (rte_flow_query(port_id, app_rules.forward_fragments[flow_i], &action[0], &count,
					   &rte_error) != 0) {
				DOCA_LOG_ERR("query failed, error=%s", rte_error.message);
			} else {
				if (flow_i < 2)
					DOCA_LOG_DBG("Port %d IPv4 fragments - %lu", port_id, count.hits);
				else
					DOCA_LOG_DBG("Port %d IPv6 fragments - %lu", port_id, count.hits);
				total_fragments += count.hits;
				total_ingress += count.hits;
			}
		}
	}
	DOCA_LOG_DBG("-------------------------------------------");
	DOCA_LOG_DBG("TOTAL INGRESS TRAFFIC:%lu", total_ingress);
	DOCA_LOG_DBG("TOTAL RSS TRAFFIC:%lu", total_rss);
	DOCA_LOG_DBG("TOTAL EGRESS TRAFFIC:%lu", total_egress);
	DOCA_LOG_DBG("TOTAL INGRESS NON_L4 TRAFFIC:%lu", total_ingress_non_l4);
	if (app_rules.forward_fragments[0] != NULL)
		DOCA_LOG_DBG("TOTAL FRAGMENTED TRAFFIC:%lu", total_fragments);
	DOCA_LOG_DBG("TOTAL DROPPED TRAFFIC:%lu", total_dropped);
}
