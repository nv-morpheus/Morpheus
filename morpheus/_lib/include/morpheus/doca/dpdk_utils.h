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

#ifndef COMMON_DPDK_UTILS_H_
#define COMMON_DPDK_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <rte_mbuf.h>
#include <rte_flow.h>

#include "morpheus/doca/offload_rules.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RX_RING_SIZE 1024
#define TX_RING_SIZE 1024
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250

void dpdk_init(int argc, char **argv);

void dpdk_fini();

void dpdk_queues_and_ports_init(struct application_dpdk_config *app_dpdk_config);

void dpdk_queues_and_ports_fini(struct application_dpdk_config *app_dpdk_config);

void print_header_info(const struct rte_mbuf *packet, const bool l2, const bool l3, const bool l4);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMMON_DPDK_UTILS_H_ */
