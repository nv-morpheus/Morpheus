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
#ifndef GPU_INIT_H
#define GPU_INIT_H

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/* Disable DPDK warnings */
#define gnu_printf printf

#include <rte_gpudev.h>

#ifdef __cplusplus
extern "C" {
#endif
#define GPU_PAGE_SIZE    (1UL << 16)
#define DEFAULT_MBUF_DATAROOM 2048
#define DEF_NB_MBUF 8192
#define MAX_BURSTS_X_QUEUE 64

struct gpu_pipeline {
	int gpu_id;
	bool gpu_support;
	bool is_host_mem;
	cudaStream_t c_stream;
	struct rte_pktmbuf_extmem ext_mem;
	struct rte_gpu_comm_list *comm_list;
};

#define CUDA_ERROR_CHECK(stmt)                                                                                         \
	do {                                                                                                           \
		cudaError_t result = (stmt);                                                                           \
		if (cudaSuccess != result) {                                                                           \
			fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,                           \
				cudaGetErrorString(result));                                                           \
		}                                                                                                      \
		assert(cudaSuccess == result);                                                                         \
	} while (0)

void gpu_init(struct gpu_pipeline *pipe);

void gpu_fini(struct gpu_pipeline *pipe);

struct rte_mempool *allocate_mempool_gpu(struct gpu_pipeline *pipe, const uint32_t mbuf_mem_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPU_INIT_H_ */
