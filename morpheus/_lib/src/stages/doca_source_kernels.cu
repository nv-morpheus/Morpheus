// #include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.h"
#include <doca_gpu_device.cuh>
#include <stdio.h>

__global__ void _kernel_receive_persistent(
  doca_gpu_rxq_info*      rxq_info,
  doca_gpu_semaphore_in*  sem_in,
  uint32_t                sem_count
)
{
  uint16_t const packet_count_rx_max = 2048;
  uint64_t const timeout_ns = 5000000; // 5ms

	__shared__ uint32_t  packet_count;
  __shared__ bool      exit_flag;

	uintptr_t packet_address;
  uint32_t  sem_idx = 0;

  if (threadIdx.x == 0){
    DOCA_GPU_VOLATILE(exit_flag) = false;
  }
  __syncthreads();

  while (DOCA_GPU_VOLATILE(exit_flag) == false)
  {
    // ===== CHECK ================================================================================

    __shared__ doca_gpu_semaphore_status  sem_status;

    // auto ret =
    doca_gpu_device_semaphore_get_value(
      sem_in + sem_idx,
      &sem_status,
      NULL,
      NULL
    );

    // if (threadIdx.x == 0) {
    //   printf("kernel: sem status %d\n", sem_status);
    // }

    __syncthreads();

    // ===== RECEIVE ==============================================================================

    DOCA_GPU_VOLATILE(packet_count) = 0;

    // if (threadIdx.x == 0) {
    //   printf("kernel: try receive for sem %d.\n", sem_idx);
    // }

    __syncthreads();

    auto ret = doca_gpu_device_receive_block(
      rxq_info,
      packet_count_rx_max,
      timeout_ns,
      sem_in + sem_idx,
      true,
      &packet_count,
      &packet_address
    );

    __threadfence();
    __syncthreads();

    if (ret != DOCA_SUCCESS) {
      doca_gpu_device_semaphore_update(
        &(sem_in[sem_idx]),
        DOCA_GPU_SEM_STATUS_ERROR,
        packet_count,
        packet_address
      );

      DOCA_GPU_VOLATILE(exit_flag) == true;
      continue;
    }

    if (packet_count <= 0)
    {
      continue;
    }

    if (threadIdx.x == 0) {
      printf("kernel: %d packet(s) received for sem %d.\n", packet_count, sem_idx);
    }

    sem_idx = (sem_idx + 1) % sem_count;

    __syncthreads();

    // ===== PROCESS ==============================================================================

    __shared__ uint32_t                   stride_start_idx;

    if (threadIdx.x == 0) {
			stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
        &(rxq_info->comm_buf),
        packet_address
      );

      // printf("kernel: stride: %d\n", stride_start_idx);
		}

    __syncthreads();

    /* Packet processing */
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

        // auto src_ip = reinterpret_cast<uint8_t*>(&packet_l4->src_port);
        // auto dst_ip = reinterpret_cast<uint8_t*>(&packet_l4->dst_port);

        // printf(
        //   "src: %d.%d.%d.%d / dst: %d.%d.%d.%d\n",
        //   src_ip[0], src_ip[1], src_ip[2], src_ip[3],
        //   dst_ip[0], dst_ip[1], dst_ip[2], dst_ip[3]
        // );

        uint32_t tmp =
            ((uint16_t*)packet_l3)[0] + ((uint16_t*)packet_l3)[1] + ((uint16_t*)packet_l3)[2] +
            ((uint16_t*)packet_l3)[3] + ((uint16_t*)packet_l3)[4] + //((uint16_t*)packet_l3)[5] +
            ((uint16_t*)packet_l3)[6] + ((uint16_t*)packet_l3)[7] + ((uint16_t*)packet_l3)[8] +
            ((uint16_t*)packet_l3)[9];

        uint16_t checksum = ~((uint16_t)(tmp & 0xFFFF) + (uint16_t)(tmp >> 16));

        if (packet_l3->hdr_checksum == checksum) {
          printf("checksum: %d / %d\n", packet_l3->hdr_checksum, checksum);
        }

        // printf(
        //   "IP: %x / %x len: %d, TCP: %d / %d dtoff: %x ih3 len: %d, pld: %x %x %x %x\n",
        //   packet_l3->src_addr,
        //   packet_l3->dst_addr,
        //   BYTE_SWAP16(packet_l3->total_length),
        //   BYTE_SWAP16(packet_l4->src_port),
        //   BYTE_SWAP16(packet_l4->dst_port),
        //   packet_l4->dt_off,
        //   (uint8_t)((packet_l3->version_ihl & RTE_IPV4_HDR_IHL_MASK) * RTE_IPV4_IHL_MULTIPLIER),
        //   packet_payload[0],
        //   packet_payload[1],
        //   packet_payload[2],
        //   packet_payload[3]
        // );
    }

    __syncthreads();

    doca_gpu_device_semaphore_update_status(
      sem_in + sem_idx,
      DOCA_GPU_SEM_STATUS_DONE
    );
  }

  printf("kernel: exiting\n");
}

void doca_receive_persistent(
  doca_gpu_rxq_info*      rxq_info,
  doca_gpu_semaphore_in*  sem_in,
  uint32_t                sem_count,
  cudaStream_t stream
)
{
  _kernel_receive_persistent<<<1, 512, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count
  );
}
