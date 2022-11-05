// #include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.h"
#include <doca_gpu_device.cuh>
#include <stdio.h>
#include <cuda/atomic>
#include <cuda/std/chrono>


__global__ void _doca_packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag
)
{
  if (threadIdx.x == 0) {
    printf("kernel receive: started\n");
  }

  uint16_t const packet_count_rx_max = 2048;
  uint64_t const timeout_ns = 50000000; // 5ms

	__shared__ uint32_t packet_count;

	uintptr_t packet_address;

  __syncthreads();

  while (*exit_flag == false)
  {
    // ===== WAIT FOR FREE SEM ====================================================================

    __shared__ doca_gpu_semaphore_status sem_status;
    __shared__ bool found_ready;

    if (threadIdx.x == 0)
    {
      found_ready = false;
      bool first_pass = true;

      while (*exit_flag == false)
      {
        auto ret = doca_gpu_device_semaphore_get_value(
          sem_in + *sem_idx,
          &sem_status,
          nullptr,
          nullptr
        );

        if (ret != DOCA_SUCCESS) {
          *exit_flag == true;
          continue;
        }

        if (sem_status == DOCA_GPU_SEM_STATUS_FREE) {
          break;
        }

        if (sem_status == DOCA_GPU_SEM_STATUS_READY)
        {
          found_ready = true;
          break;
        }

        if (first_pass) {
          first_pass = false;
          printf("kernel receive: waiting on sem %d to become free\n", *sem_idx);
        }
      }
    }
    
    __syncthreads();

    if (found_ready)
    {
      return;
    }

    // ===== RECEIVE TO FREE SEM ==================================================================

    DOCA_GPU_VOLATILE(packet_count) = 0;

    auto ret = doca_gpu_device_receive_block(
      rxq_info,
      packet_count_rx_max,
      timeout_ns,
      sem_in + *sem_idx,
      true,
      &packet_count,
      &packet_address
    );

    __threadfence();
    __syncthreads();

    if (ret != DOCA_SUCCESS) {
      printf("kernel receive: setting sem %d to error\n", *sem_idx);
      doca_gpu_device_semaphore_update(
        sem_in + *sem_idx,
        DOCA_GPU_SEM_STATUS_ERROR,
        packet_count,
        packet_address
      );

      *exit_flag == true;
      continue;
    }

    if (packet_count <= 0)
    {
      continue;
    }

    if (threadIdx.x == 0) {
      printf("kernel receive: setting sem %d to ready\n", *sem_idx);
      doca_gpu_device_semaphore_update_status(
        sem_in + *sem_idx,
        DOCA_GPU_SEM_STATUS_READY
      );
    }
    __threadfence();

    if (threadIdx.x == 0) {
      printf("kernel receive: %d packet(s) received for sem %d\n", packet_count, *sem_idx);
    }
    
    if (threadIdx.x == 0)
    {
      *sem_idx = (*sem_idx + 1) % sem_count;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    printf("kernel receive: end\n");
  }
}

__global__ void _doca_packet_count_kernel(
  doca_gpu_rxq_info*                                rxq_info,
  doca_gpu_semaphore_in*                            sem_in,
  uint32_t                                          sem_count,
  uint32_t*                                         sem_idx_begin,
  uint32_t*                                         sem_idx_end,
  uint32_t*                                         packet_count_out,
  uint32_t*                                         packets_size_out,
  cuda::atomic<bool, cuda::thread_scope_system>*    exit_flag
)
{
  if (threadIdx.x == 0) {
    printf("kernel count: started\n");
  }

  *packet_count_out = 0;

  __shared__ doca_gpu_semaphore_status sem_status;
	__shared__ uint32_t  packet_count;

	uintptr_t packet_address;

  auto sem_idx = *sem_idx_begin;

  __syncthreads();

  // ===== WAIT FOR READY SEM ===================================================================

  bool first_pass = true;

  bool should_count_next_sem = true;

  while (*exit_flag == false)
  {
    while (*exit_flag == false)
    {
      auto ret = doca_gpu_device_semaphore_get_value(
        sem_in + sem_idx,
        &sem_status,
        &packet_count,
        &packet_address
      );

      if (ret != DOCA_SUCCESS) {
        *exit_flag = true;
        return;
      }

      if (sem_status == DOCA_GPU_SEM_STATUS_READY) {
        break;
      }

      if (sem_status == DOCA_GPU_SEM_STATUS_HOLD) {
        should_count_next_sem = false;
        break;
      }

      if (threadIdx.x == 0 and first_pass) {
        printf("kernel count: waiting on sem %d to become ready\n", sem_idx);
      }

      first_pass = false;
    }

    __syncthreads();

    if (not should_count_next_sem)
    {
      break;
    }

    if (threadIdx.x == 0 and not first_pass) {
      printf("kernel count: counting sem %d\n", sem_idx);
    }

    __syncthreads();

    // ===== COUNT PACKETS IN SEM =================================================================

    for (auto packet_idx = 0; packet_idx < packet_count; packet_idx++)
    {
      // compute total packet payload size
      // atomicAdd(packets_size_out, packet_payload_length)
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
      *packet_count_out += packet_count;

      // TODO: compute packet payload size
      *packets_size_out = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
      printf("kernel count: setting sem %d to held\n", sem_idx);
      doca_gpu_device_semaphore_update_status(
        sem_in + sem_idx,
        DOCA_GPU_SEM_STATUS_HOLD
      );
    }

    __threadfence();

    sem_idx = (sem_idx + 1) % sem_count;
  }

  if(threadIdx.x == 0)
  {
    *sem_idx_end = sem_idx;
    printf("kernel count: %d packets counted for sems [%d, %d) with total payload size %d\n", *packet_count_out, *sem_idx_begin, *sem_idx_end, *packets_size_out);
  }

  if (threadIdx.x == 0)
  {
    printf("kernel count: done\n");
  }

  __syncthreads();
}

__global__ void _doca_packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx_begin,
  uint32_t*                                       sem_idx_end,
  uint32_t*                                       packet_count_out,
  uint32_t*                                       packets_size_out,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag
)
{
  if (threadIdx.x == 0) {
    printf("kernel gather: started\n");
  }

  __shared__ doca_gpu_semaphore_status sem_status;
	__shared__ uint32_t  packet_count;

	uintptr_t packet_address;

  uint32_t sem_idx = *sem_idx_begin;

  // ===== WAIT FOR HELD SEM ======================================================================

  // don't need to wait because we know which sems to process.
  // rule 1: sem at sem_idx_begin must be processed, because we wouldn't be here if there weren't at least one sem to process.
  // rule 2: all sems up to sem_idx_end (exclusive) must be processed.
  // rule 3: if sem_idx_begin == sem_idx_end, sem_idx_begin still gets processed due to rule 1.

  __shared__ uint32_t packet_offset;

  if (threadIdx.x == 0)
  {
    packet_offset = 0;
  }

  __syncthreads();

  while (*exit_flag == false)
  {
    // get sem info
    auto ret = doca_gpu_device_semaphore_get_value(
      sem_in + sem_idx,
      &sem_status,
      &packet_count,
      &packet_address
    );

    if (ret != DOCA_SUCCESS)
    {
      *exit_flag = true;
      continue;
    }

    __syncthreads();

    // copy packets to dataframe

    __shared__ uint32_t stride_start_idx;

    if (threadIdx.x == 0) {
			stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
        &(rxq_info->comm_buf),
        packet_address
      );
    }

    __syncthreads();

    for (auto packet_idx = threadIdx.x; packet_idx < packet_count; packet_idx += blockDim.x)
    {
      uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
        &(rxq_info->comm_buf),
        stride_start_idx + packet_idx
      );

      rte_ether_hdr* packet_l2;
      rte_ipv4_hdr*  packet_l3;
      rte_tcp_hdr*   packet_l4;
      uint8_t*       packet_payload;

      get_packet_tcp_headers(
        packet,
        &packet_l2,
        &packet_l3,
        &packet_l4,
        &packet_payload
      );

      auto total_length = BYTE_SWAP16(packet_l3->total_length);

      // mac address printing works
      auto src_mac = packet_l2->s_addr.addr_bytes; // 6 bytes
      auto dst_mac = packet_l2->d_addr.addr_bytes; // 6 bytes

      if (threadIdx.x == 0)
      {
        printf(
          "= mac: %02x:%02x:%02x:%02x:%02x:%02x / %02x:%02x:%02x:%02x:%02x:%02x\n",
          src_mac[0], src_mac[1], src_mac[2], src_mac[3], src_mac[4], src_mac[5],
          dst_mac[0], dst_mac[1], dst_mac[2], dst_mac[3], dst_mac[4], dst_mac[5]
        );
      }

      // ip address printing works
      auto src_address  = packet_l3->src_addr;
      auto dst_address  = packet_l3->dst_addr;
      auto src_port     = BYTE_SWAP16(packet_l4->src_port);
      auto dst_port     = BYTE_SWAP16(packet_l4->dst_port);

      auto src_address_ptr = reinterpret_cast<uint8_t*>(&src_address);
      auto dst_address_ptr = reinterpret_cast<uint8_t*>(&dst_address);

      if (threadIdx.x == 0)
      {
        printf(
          "= ip: %d.%d.%d.%d:%d / %d.%d.%d.%d:%d \n",
          src_address_ptr[0], src_address_ptr[1], src_address_ptr[2], src_address_ptr[3],
          src_port,
          dst_address_ptr[0], dst_address_ptr[1], dst_address_ptr[2], dst_address_ptr[3],
          dst_port
        );
      }

      // reverse the bytes so int64->ip string kernel works properly.

      auto src_address_rev = (src_address & 0x000000ff) << 24
                           | (src_address & 0x0000ff00) << 8
                           | (src_address & 0x00ff0000) >> 8
                           | (src_address & 0xff000000) >> 24;

      auto dst_address_rev = (dst_address & 0x000000ff) << 24
                           | (dst_address & 0x0000ff00) << 8
                           | (dst_address & 0x00ff0000) >> 8
                           | (dst_address & 0xff000000) >> 24;

      src_ip_out[packet_offset + packet_idx] = src_address_rev;
      dst_ip_out[packet_offset + packet_idx] = dst_address_rev;
    }

    __syncthreads();

    // release sem

    if (threadIdx.x == 0)
    {
      printf("kernel gather: setting sem %d to free\n", sem_idx);

      doca_gpu_device_semaphore_update_status(
        sem_in + sem_idx,
        DOCA_GPU_SEM_STATUS_FREE
      );

      packet_offset += packet_count;
    }

    __syncthreads();

    // determine if the next sem should be processed

    sem_idx = (sem_idx + 1) % sem_count;

    if (sem_idx == *sem_idx_end)
    {
      break;
    }
  }

  *sem_idx_begin = *sem_idx_end;

  if (threadIdx.x == 0) {
    printf("kernel gather: done\n");
  }

  __syncthreads();
}

void doca_packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
)
{
  _doca_packet_receive_kernel<<<1, 512, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    exit_flag
  );
}

void doca_packet_count_kernel(
  doca_gpu_rxq_info*                                rxq_info,
  doca_gpu_semaphore_in*                            sem_in,
  uint32_t                                          sem_count,
  uint32_t*                                         sem_idx_begin,
  uint32_t*                                         sem_idx_end,
  uint32_t*                                         packet_count,
  uint32_t*                                         packets_size,
  cuda::atomic<bool, cuda::thread_scope_system>*    exit_flag,
  cudaStream_t                                      stream
)
{
  _doca_packet_count_kernel<<<1, 512, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx_begin,
    sem_idx_end,
    packet_count,
    packets_size,
    exit_flag
  );
}


void doca_packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx_begin,
  uint32_t*                                       sem_idx_end,
  uint32_t*                                       packet_count,
  uint32_t*                                       packets_size,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
)
{
  _doca_packet_gather_kernel<<<1, 512, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx_begin,
    sem_idx_end,
    packet_count,
    packets_size,
    src_ip_out,
    dst_ip_out,
    exit_flag
  );
}
