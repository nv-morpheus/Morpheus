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

#include <doca_error.h>

#ifdef __cplusplus
extern "C" {
#endif
#define GPU_PAGE_SIZE (1UL << 16)		/* GPU page size */
#define DEFAULT_MBUF_DATAROOM 2048		/* Mbuf data room size */
#define DEF_NB_MBUF 8192			/* Number of mbuf's */
#define COMM_LIST_LEN 64			/* Length of communication objects list */

struct gpu_pipeline {
	int gpu_id;					/* GPU id */
	bool gpu_support;				/* Enable program GPU support */
	bool is_host_mem;				/* Allocate mbuf's mempool on GPU or device memory */
	cudaStream_t c_stream;				/* CUDA stream */
	struct rte_pktmbuf_extmem ext_mem;		/* Mbuf's mempool */
	struct rte_gpu_comm_list *comm_list;		/* Communication list between device(GPU) and host(CPU) */
};

/* CUDA result check */
#define CUDA_ERROR_CHECK(stmt)                                                                                         \
	do {                                                                                                           \
		cudaError_t result = (stmt);                                                                           \
		if (cudaSuccess != result) {                                                                           \
			fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,                           \
				cudaGetErrorString(result));                                                           \
		}                                                                                                      \
		assert(cudaSuccess == result);                                                                         \
	} while (0)

/*
 * Initialized GPU resources
 *
 * @pipe [in/out]: GPU pipeline
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t gpu_init(struct gpu_pipeline *pipe);

/*
 * GPU resources cleanup
 *
 * @pipe [in]: GPU pipeline
 */
void gpu_fini(struct gpu_pipeline *pipe);

/*
 * Create mbuf's mempool
 * Mempool could be located in the host (CPU) shared memory and register it to the GPU device or in GPU memory and
 * attach it to DPDK as an external mempool.
 *
 * @total_nb_mbufs [in]: mempool size
 * @pipe [in/out]: GPU pipeline
 * @mpool_payload [out]: mbuf's mempool
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t allocate_mempool_gpu(const uint32_t total_nb_mbufs, struct gpu_pipeline *pipe,
	struct rte_mempool **mpool_payload);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPU_INIT_H_ */
