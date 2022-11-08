#pragma once

#include <doca_flow.h>
#include <cuda/atomic>

namespace morpheus {
namespace doca {

void packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx_begin,
  uint32_t*                                       sem_idx_end,
  uint32_t*                                       packet_count,
  uint32_t                                        packet_count_threshold,
  cuda::std::chrono::duration<double>             debounce_threshold,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
);

void packet_count_kernel(
  doca_gpu_rxq_info*                                rxq_info,
  doca_gpu_semaphore_in*                            sem_in,
  uint32_t                                          sem_count,
  uint32_t*                                         sem_idx_begin,
  uint32_t*                                         sem_idx_end,
  uint32_t*                                         packet_count,
  uint32_t*                                         packets_size,
  cuda::atomic<bool, cuda::thread_scope_system>*    exit_flag,
  cudaStream_t                                      stream
);


void packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx_begin,
  uint32_t*                                       sem_idx_end,
  uint32_t*                                       packet_count,
  uint32_t*                                       packets_size,
  int64_t*                                        src_mac_out,
  int64_t*                                        dst_mac_out,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  uint16_t*                                       src_port_out,
  uint16_t*                                       dst_port_out,
  cuda::atomic<bool, cuda::thread_scope_system>*  exit_flag,
  cudaStream_t                                    stream
);

}
}
