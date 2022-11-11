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
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_gpudev.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>

#include <doca_log.h>

#include "morpheus/doca/gpu_init.h"

DOCA_LOG_REGISTER(GPU_INIT);

doca_error_t
allocate_mempool_gpu(const uint32_t total_nb_mbufs, struct gpu_pipeline *pipe, struct rte_mempool **mpool_payload)
{
	int result, gpu_id = pipe->gpu_id;

	/* Set mbuf element size, including size of data buffer and headroom */
	pipe->ext_mem.elt_size = DEFAULT_MBUF_DATAROOM + RTE_PKTMBUF_HEADROOM;
	pipe->ext_mem.buf_len = RTE_ALIGN_CEIL(total_nb_mbufs * pipe->ext_mem.elt_size, GPU_PAGE_SIZE);

	/* Packets mempool can be located in host (CPU) memory or in the device (GPU) memory */
	if (pipe->is_host_mem) {
		pipe->ext_mem.buf_ptr = rte_malloc(NULL, pipe->ext_mem.buf_len, 0);
		if (pipe->ext_mem.buf_ptr == NULL) {
			DOCA_LOG_ERR("Could not allocate CPU DPDK memory");
			return DOCA_ERROR_NO_MEMORY;
		}

		result = rte_gpu_mem_register(gpu_id, pipe->ext_mem.buf_len, pipe->ext_mem.buf_ptr);
		if (result < 0) {
			DOCA_LOG_ERR("Unable to gpudev register addr %p, result %d", pipe->ext_mem.buf_ptr, result);
			rte_free(pipe->ext_mem.buf_ptr);
			return DOCA_ERROR_DRIVER;
		}
	} else {
		pipe->ext_mem.buf_iova = RTE_BAD_IOVA;
		pipe->ext_mem.buf_ptr = rte_gpu_mem_alloc(gpu_id, pipe->ext_mem.buf_len, 0);
		if (pipe->ext_mem.buf_ptr == NULL) {
			DOCA_LOG_ERR("Could not allocate GPU device memory");
			return DOCA_ERROR_NO_MEMORY;
		}

		result = rte_extmem_register(pipe->ext_mem.buf_ptr, pipe->ext_mem.buf_len, NULL, pipe->ext_mem.buf_iova,
					  GPU_PAGE_SIZE);
		if (result < 0) {
			DOCA_LOG_ERR("Unable to register addr %p, result %d", pipe->ext_mem.buf_ptr, result);
			rte_gpu_mem_free(gpu_id, pipe->ext_mem.buf_ptr);
			return DOCA_ERROR_DRIVER;
		}

	}
	/* Create mempool with attaching external memory buffer */
	*mpool_payload = rte_pktmbuf_pool_create_extbuf("payload_mpool", total_nb_mbufs, 0, 0, pipe->ext_mem.elt_size,
						       rte_socket_id(), &pipe->ext_mem, 1);
	if (*mpool_payload == NULL) {
		DOCA_LOG_ERR("Could not create EXT memory mempool");
		if (pipe->is_host_mem)
			rte_free(pipe->ext_mem.buf_ptr);
		else
			rte_gpu_mem_free(gpu_id, pipe->ext_mem.buf_ptr);
		return DOCA_ERROR_DRIVER;
	}

	return DOCA_SUCCESS;
}

doca_error_t
gpu_init(struct gpu_pipeline *pipe)
{
	struct rte_gpu_info ginfo;
	int gpu_idx = 0;
	uint32_t nb_gpus, gpu_id = pipe->gpu_id;

	/* Prevent any useless profiling */
	cudaProfilerStop();

	/* Set CUDA GPU device id and init CUDA stream */
	cudaSetDevice(gpu_id);
	CUDA_ERROR_CHECK(cudaStreamCreateWithFlags(&(pipe->c_stream), cudaStreamDefault));

	/* Find the number of available GPU devices */
	nb_gpus = rte_gpu_count_avail();

	/* Print info of the GPU device */
	DOCA_LOG_INFO("DPDK found %d GPUs:", nb_gpus);
	RTE_GPU_FOREACH(gpu_idx)
	{
		if (rte_gpu_info_get(gpu_idx, &ginfo)) {
			DOCA_LOG_ERR("Failed to get gpu device info");
			return DOCA_ERROR_DRIVER;
		}

		DOCA_LOG_INFO("\tGPU ID %d", ginfo.dev_id);
		DOCA_LOG_INFO("\t\tParent ID %d GPU Bus ID %s NUMA node %d Tot memory %.02f MB, Tot processors %d",
			      ginfo.parent, ginfo.name, ginfo.numa_node,
			      (((float)ginfo.total_memory) / (float)1024) / (float)1024, ginfo.processor_count);
		break;
	}

	if (nb_gpus == 0 || nb_gpus < gpu_id) {
		DOCA_LOG_ERR("Error nb_gpus %d gpu_id %d", nb_gpus, gpu_id);
		return DOCA_ERROR_DRIVER;
	}

	/* CommList will be used to communicate between DPU-GPU */
	pipe->comm_list = rte_gpu_comm_create_list(pipe->gpu_id, COMM_LIST_LEN);
	if (pipe->comm_list == NULL) {
		DOCA_LOG_ERR("Failed to create communication list");
		return DOCA_ERROR_DRIVER;
	}
	return DOCA_SUCCESS;
}

void
gpu_fini(struct gpu_pipeline *pipe)
{
	int result;
	struct rte_pktmbuf_extmem *ext_mem = &pipe->ext_mem;

	/* Destroy CUDA stream */
	cudaStreamDestroy(pipe->c_stream);

	/* Destroy DPU-GPU communication list */
	result = rte_gpu_comm_destroy_list(pipe->comm_list, COMM_LIST_LEN);
	if (result != 0)
		DOCA_LOG_ERR("Communication list release failed, returned error %d", result);

	if (pipe->is_host_mem) {
		result = rte_gpu_mem_unregister(pipe->gpu_id, ext_mem->buf_ptr);
		if (result < 0)
			DOCA_LOG_ERR("Unable to unregister addr %p, result %d", ext_mem->buf_ptr, result);
	} else {
		result = rte_gpu_mem_free(pipe->gpu_id, ext_mem->buf_ptr);
		if (result < 0)
			DOCA_LOG_ERR("Failed to free GPU memory addr %p, result %d ", ext_mem->buf_ptr, result);
	}
}
