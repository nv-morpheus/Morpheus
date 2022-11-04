#pragma once

#include <doca_flow.h>
#include <cuda/atomic>

void doca_packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
);

void doca_packet_count_kernel(
  doca_gpu_rxq_info*                                rxq_info,
  doca_gpu_semaphore_in*                            sem_in,
  uint32_t                                          sem_count,
  uint32_t*                                         sem_idx_begin,
  uint32_t*                                         sem_idx_end,
  cuda::std::chrono::duration<int64_t>              debounce_max,
  uint32_t                                          packet_count_max,
  uint32_t*                                         packet_count,
  uint32_t*                                         packets_size,
  cuda::atomic<bool, cuda::thread_scope_system>*    exit_flag,
  cudaStream_t                                      stream
);


void doca_packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx_begin,
  uint32_t*                                       sem_idx_end,
  uint32_t*                                       packet_count,
  uint32_t*                                       packets_size,
  uint32_t*                                       src_ip_out,
  uint32_t*                                       dst_ip_out,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
);
