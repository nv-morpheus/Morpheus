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

#ifndef DOCA_GPU_PACKET_PROCESSING_H
#define DOCA_GPU_PACKET_PROCESSING_H

#include <stdio.h>
#include <bsd/string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"
#include <signal.h>

#include <rte_ethdev.h>

#include <doca_log.h>
#include <doca_version.h>
#include <doca_gpu.h>
#include <doca_argp.h>
#include <doca_flow.h>

#include "morpheus/doca/dpdk_utils.h"

#define BE_IPV4_ADDR(a, b, c, d) (RTE_BE32((a << 24) + (b << 16) + (c << 8) + d))

#define MAX_PORT_STR_LEN		128  /* Maximal length of port name */
#define MAX_PCI_ADDRESS_LEN		32U
#define MAX_QUEUES			4
#define INFERENCE_GOOD_PACKETS_ITEMS	32
#define MAX_INFERENCE_ITEMS		8
#define MAX_PKT_PAYLOAD			2048
#define MAX_PKT_HTTP_PAYLOAD		8192
#define INFERENCE_CLEANUP_THRESHOLD (INFERENCE_ITEMS_X_RXQ/4)

enum workload_mode {
	WORKLOAD_PERSISTENT = 0,
	WORKLOAD_SINGLE,
};

enum processing_mode {
	PROCESSING_IP_CHECKSUM = 0,
	PROCESSING_INFERENCE_HTTP,
	PROCESSING_FORWARD,
};

enum receive_mode {
	RECEIVE_CPU = 0,
	RECEIVE_GPU_DEDICATED,
	RECEIVE_GPU_PROXY,
};

#define IP_ADD_0 16
#define IP_ADD_1 0
#define IP_ADD_2 0
#define IP_ADD_3 4

#define MEM_ALIGN_SZ 4096

#define WARP_FULL_MASK 0xffffffff
#define WARP_SIZE 32

#define CUDA_BLOCKS 4
#define CUDA_THREADS 512
#define DEBUG_PRINT 0

#define BYTE_SWAP16(v) \
	((((uint16_t)(v) & UINT16_C(0x00ff)) << 8) | (((uint16_t)(v) & UINT16_C(0xff00)) >> 8))

/**
 * Holds all configuration for the application
 */
struct app_gpu_inline_cfg {
	char gpu_pcie_addr[MAX_PCI_ADDRESS_LEN];
	uint32_t nic_port;
	uint8_t processing;
	uint8_t queue_num;
	uint8_t receive_mode;
	uint8_t workload_mode;
};

struct proxy_sem_out {
	uint32_t tot_pkts;
};

/*
data_dict = {
	'time': packet.frame_info.time,
	'time_epoch': packet.frame_info.time_epoch,
	'eth_src': packet.eth.src,
	'eth_dst': packet.eth.dst,
	'ip_src': packet.ip.src,
	'ip_dst': packet.ip.dst,
	'tcp_srcport': packet.tcp.srcport,
	'tcp_dstport': packet.tcp.dstport,
	'data': data,
	}
*/
struct inference_entry {
	uint32_t ip_src;
	uint32_t ip_dst;
	uint16_t tcp_src_port;
	uint16_t tcp_dst_port;
	uint16_t l5_size;
	uint8_t payload[MAX_PKT_HTTP_PAYLOAD];
};

struct inference_item {
	struct inference_entry entry[MAX_INFERENCE_ITEMS];
};

struct inference_sem_out {
	uint32_t err_pkts;
	uint32_t http_pkts;
	uint32_t get_pkts;
	uint32_t post_pkts;
	uint32_t head_pkts;
};

struct queue_items {
	struct doca_gpu_rxq_info *rxq_info_cpu;
	struct doca_gpu_rxq_info *rxq_info_gpu;
	struct doca_flow_pipe *rxq_pipe;

	struct doca_gpu_semaphore *sem_proxy_h;
	struct doca_gpu_semaphore_in *sem_proxy_in_gpu;
	struct doca_gpu_semaphore_in *sem_proxy_in_cpu;
	struct proxy_sem_out *sem_proxy_info_gpu;
	struct proxy_sem_out *sem_proxy_info_cpu;

	struct doca_gpu_txq_info *txq_info_cpu;
	struct doca_gpu_txq_info *txq_info_gpu;
};

#if __cplusplus
extern "C" {
#endif

void register_gpu_params(void);
struct doca_flow_port * init_doca_flow(uint8_t port_id, uint8_t rxq_num, struct application_dpdk_config *dpdk_config);
struct doca_flow_pipe * build_rxq_pipe(
    uint16_t port_id,
    struct doca_flow_port *port,
    uint32_t source_ip_filter,
    uint16_t dpdk_rxq_idx);
doca_error_t destroy_doca_flow(uint8_t port_id, uint8_t rxq_num, struct doca_flow_pipe **rxq_pipe);

doca_error_t kernel_proxy_persistent(cudaStream_t stream, uint32_t *exit_cond, uint16_t max_rx_pkts, uint64_t timeout_ns, uint32_t sem_proxy_num,
					struct queue_items *rxq_h_0, struct queue_items *rxq_h_1, struct queue_items *rxq_h_2, struct queue_items *rxq_h_3,
					struct doca_gpu_semaphore_in *sem_inference_in, uint32_t sem_inference_num, bool receive, bool send);

doca_error_t kernel_receive_persistent(cudaStream_t stream, uint32_t *exit_cond, uint16_t max_rx_pkts, uint64_t timeout_ns, uint32_t sem_proxy_num, uint8_t sem_check,
				       struct queue_items *rxq_h_0, struct queue_items *rxq_h_1, struct queue_items *rxq_h_2, struct queue_items *rxq_h_3);

doca_error_t kernel_inference_http_persistent(cudaStream_t stream, uint32_t *exit_cond, struct doca_gpu_semaphore_in *sem_inference, uint32_t sem_inference_num, struct inference_sem_out* sem_out);

doca_error_t kernel_proxy_single(cudaStream_t stream,
				struct queue_items *rxq_h_0, struct queue_items *rxq_h_1, struct queue_items *rxq_h_2, struct queue_items *rxq_h_3,
				uint32_t sem_idx_0, uint32_t sem_idx_1, uint32_t sem_idx_2, uint32_t sem_idx_3);

__device__ __forceinline__ unsigned long long _gputimestamp()
{
	unsigned long long globaltimer;
	// 64-bit GPU global nanosecond timer
	asm volatile("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
	return globaltimer;
}

__forceinline__ __host__ __device__ int
get_packet_tcp_headers(uint8_t* pkt, struct rte_ether_hdr **l2_hdr, struct rte_ipv4_hdr **l3_hdr, struct rte_tcp_hdr **l4_hdr, uint8_t **l5_pld)
{
	(*l2_hdr) = (struct rte_ether_hdr *) pkt;
	(*l3_hdr) = (struct rte_ipv4_hdr *) (((uint8_t*) (*l2_hdr)) + RTE_ETHER_HDR_LEN);
	(*l4_hdr) = (struct rte_tcp_hdr *) (((uint8_t*)(*l3_hdr)) + (uint8_t)(((*l3_hdr)->version_ihl & RTE_IPV4_HDR_IHL_MASK) * RTE_IPV4_IHL_MULTIPLIER));
	(*l5_pld) = ((uint8_t*)(*l4_hdr)) + (uint8_t)((*l4_hdr)->dt_off * sizeof(int));

	return 0;
}

#if __cplusplus
}
#endif

#endif /* DOCA_GPU_PP_COMMON_H */
