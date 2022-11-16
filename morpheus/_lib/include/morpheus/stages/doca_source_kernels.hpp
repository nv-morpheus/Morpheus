#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <doca_flow.h>
#include <cuda/atomic>
#include <memory>

namespace morpheus {
namespace doca {

std::unique_ptr<cudf::column> integers_to_mac(
  cudf::column_view const& integers,
  rmm::cuda_stream_view stream = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

void packet_receive_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint32_t*                                       packet_count,
  uint32_t*                                       packet_data_size,
  cudaStream_t                                    stream
);

void packet_gather_kernel(
  doca_gpu_rxq_info*                              rxq_info,
  doca_gpu_semaphore_in*                          sem_in,
  uint32_t                                        sem_count,
  uint32_t*                                       sem_idx,
  uint64_t*                                       timestamp_out,
  int64_t*                                        src_mac_out,
  int64_t*                                        dst_mac_out,
  int64_t*                                        src_ip_out,
  int64_t*                                        dst_ip_out,
  uint16_t*                                       src_port_out,
  uint16_t*                                       dst_port_out,
  int32_t*                                        data_offsets_out,
  char*                                           data_out,
  cudaStream_t                                    stream
);

}
}
