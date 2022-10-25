// #include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.h"
// #include <doca_gpu_device.cuh>
#include <stdio.h>

__global__ void _kernel_receive_persistent(
  doca_gpu_rxq_info*      rxq_info,
  doca_gpu_semaphore_in*  sem_in,
  uint32_t                sem_count
)
{
  if (blockIdx.x != 0) {
    return;
  }

  printf("kernel: Hello World!\n");

	__shared__ uint32_t  packet_count;
	__shared__ uintptr_t packet_address;

  // for (auto sem_idx = 0; sem_idx < sem_count;)
  // {
  //   uint16_t packet_count_rx_max = 2048;
  //   uint64_t timeout_ns = 5000000; // 5ms

  //   auto ret = doca_gpu_device_receive_block(
  //     rxq_info,
  //     packet_count_rx_max,
  //     timeout_ns,
  //     &(sem_in[sem_idx]),
  //     true,
  //     &packet_count,
  //     &packet_address
  //   );

  //   if (ret != DOCA_SUCCESS) {
  //     printf("kernel: packet receive failed.");
  //     // TODO: determine if this should be called on all threads or not.
	// 		doca_gpu_device_semaphore_update(
  //       &(sem_in[sem_idx]),
  //       DOCA_GPU_SEM_STATUS_ERROR,
  //       packet_count,
  //       packet_address
  //     );
  //   }

  //   auto should_exit = __any_sync(0xffffffff, ret != DOCA_SUCCESS);

  //   if (should_exit) {
  //     printf("kernel: exiting due to error.\n", packet_count);
  //     return;
  //   }

  //   printf("kernel: %d packet(s) recieved.\n", packet_count);

  //   if (packet_count > 0) {
  //     sem_idx++;
  //   }
  // }
}

void doca_receive_persistent(
  doca_gpu_rxq_info*      rxq_info,
  doca_gpu_semaphore_in*  sem_in,
  uint32_t                sem_count,
  cudaStream_t stream
)
{
  _kernel_receive_persistent<<<1, 1, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count
  );
}
