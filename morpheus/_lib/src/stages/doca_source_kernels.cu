// #include "morpheus/doca/doca_context.hpp"

#include "morpheus/doca/common.h"
#include <doca_gpu_device.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cuda/std/chrono>
#include <memory>
#include <stdio.h>
#include <thrust/iterator/constant_iterator.h>
#include <cub/cub.cuh>

__device__ char to_hex_16(uint8_t value)
{
    return "0123456789ABCDEF"[value];
}

__device__ int64_t mac_bytes_to_int64(uint8_t* mac)
{
  return static_cast<uint64_t>(mac[0]) << 40
        | static_cast<uint64_t>(mac[1]) << 32
        | static_cast<uint32_t>(mac[2]) << 24
        | static_cast<uint32_t>(mac[3]) << 16
        | static_cast<uint32_t>(mac[4]) << 8
        | static_cast<uint32_t>(mac[5]);
}

__device__ int64_t mac_int64_to_chars(int64_t mac, char* out)
{
  uint8_t mac_0 = (mac >> 40) & (0xFF);
  out[0]  = to_hex_16(mac_0 / 16);
  out[1]  = to_hex_16(mac_0 % 16);
  out[2]  = ':';

  uint8_t mac_1 = (mac >> 32) & (0xFF);
  out[3]  = to_hex_16(mac_1 / 16);
  out[4]  = to_hex_16(mac_1 % 16);
  out[5]  = ':';

  uint8_t mac_2 = (mac >> 24) & (0xFF);
  out[6]  = to_hex_16(mac_2 / 16);
  out[7]  = to_hex_16(mac_2 % 16);
  out[8]  = ':';

  uint8_t mac_3 = (mac >> 16) & (0xFF);
  out[9]  = to_hex_16(mac_3 / 16);
  out[10] = to_hex_16(mac_3 % 16);
  out[11] = ':';

  uint8_t mac_4 = (mac >> 8) & (0xFF);
  out[12] = to_hex_16(mac_4 / 16);
  out[13] = to_hex_16(mac_4 % 16);
  out[14] = ':';

  uint8_t mac_5 = (mac >> 0) & (0xFF);
  out[15] = to_hex_16(mac_5 / 16);
  out[16] = to_hex_16(mac_5 % 16);
}

uint32_t const PACKETS_PER_THREAD = 4;
uint32_t const THREADS_PER_BLOCK = 512;
uint32_t const PACKETS_PER_BLOCK = PACKETS_PER_THREAD * THREADS_PER_BLOCK;
// uint32_t const PACKET_RX_TIMEOUT_NS = 5000000; // 5ms
uint32_t const PACKET_RX_TIMEOUT_NS = 50000000; // 50ms

__global__ void _packet_receive_kernel(
  doca_gpu_rxq_info*     rxq_info,
  doca_gpu_semaphore_in* sem_in,
  int32_t                sem_count,
  int32_t*               sem_idx,
  int32_t*               packet_count_out,
  int32_t*               packet_data_size_out
)
{
  if (threadIdx.x == 0)
  {
    *packet_count_out = 0;
    *packet_data_size_out = 0;
  }
  
  __shared__ uint32_t packet_count;
  __shared__ doca_gpu_semaphore_status sem_status;
  
  uintptr_t packet_address;

  if (threadIdx.x == 0)
  {
    while (true)
    {
      auto ret = doca_gpu_device_semaphore_get_value(
        sem_in + *sem_idx,
        &sem_status,
        nullptr,
        nullptr
      );

      if (sem_status == DOCA_GPU_SEM_STATUS_FREE)
      {
        break;
      }
    }
  }

  __syncthreads();

  DOCA_GPU_VOLATILE(packet_count) = 0;

  __syncthreads();

  auto ret = doca_gpu_device_receive_block(
    rxq_info,
    PACKETS_PER_BLOCK,
    PACKET_RX_TIMEOUT_NS,
    nullptr,
    false,
    &packet_count,
    &packet_address
  );

  __threadfence();
  __syncthreads();

  if (packet_count == 0) {
    return;
  }

  __shared__ uint32_t stride_start_idx;

  if (threadIdx.x == 0) {
    *packet_count_out = packet_count;
    stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
      &(rxq_info->comm_buf),
      packet_address
    );
  }

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto data_size = total_length - static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));

    atomicAdd(packet_data_size_out, data_size);

    // printf("packet_idx(%d) data_size(%d) atom\n", packet_idx, data_size);
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    doca_gpu_device_semaphore_update(
      sem_in + *sem_idx,
      DOCA_GPU_SEM_STATUS_HOLD,
      packet_count,
      packet_address
    );
    // doca_gpu_device_semaphore_update_status(
    //   sem_in + *sem_idx,
    //   DOCA_GPU_SEM_STATUS_HOLD
    // );
  }

  __threadfence();
  __syncthreads();
}

__device__ uint32_t tcp_parse_timestamp(rte_tcp_hdr const *tcp)
{
// 	const uint8_t *tcp_opt = (typeof(tcp_opt))tcp + RTE_TCP_MIN_HDR_LEN;
// 	const uint8_t *tcp_data = (typeof(tcp_data))tcp + static_cast<int32_t>(tcp->dt_off * sizeof(int32_t));
// 	int ret;
// parse:
// 	switch (static_cast<rte_tcp_opt>(tcp_opt[0])) {
// 	default:
// 		// SFT_TCP_LOG(ERR, "invalid TCP option %u\n", tcp_opt[0]);
// 		ret = -1;
// 		goto out;
// 	case RTE_TCP_OPT_END:
// 		ret = 0;
// 		goto out;
// 	case RTE_TCP_OPT_NOP:
// 		tcp_opt++;
// 		break;
// 	case RTE_TCP_OPT_MSS:
// 		// sender->max_seg_size = ((uint16_t)tcp_opt[2]) << 8 | tcp_opt[3];
// 		tcp_opt += tcp_opt[1];
// 		break;
// 	case RTE_TCP_OPT_WND_SCALE:
// 		// sender->wnd_scale = tcp_opt[2];
// 		tcp_opt += tcp_opt[1];
// 		break;
// 	case RTE_TCP_OPT_SACK_PERMITTED:
// 		// sender->sack_permitted = 1;
// 		tcp_opt += tcp_opt[1];
// 		break;
// 	case RTE_TCP_OPT_SACK:
// 		// SFT_TCP_LOG(ERR, "TCP option SACK not implemented\n");
// 		ret = -1;
// 		break;
// 	case RTE_TCP_OPT_TIMESTAMP:
//     return tcp_opt[1];
// 		// tcp_opt += tcp_opt[1];
//     return  (static_cast<uint32_t>(tcp_opt[2]) << 24)
//           | (static_cast<uint32_t>(tcp_opt[3]) << 16)
//           | (static_cast<uint32_t>(tcp_opt[4]) << 8)
//           | (static_cast<uint32_t>(tcp_opt[5]));
// 		break;
// 	}
// 	if (tcp_opt == tcp_data) {
// 		ret = 0;
// 		goto out;
// 	} else if (tcp_opt < tcp_data) {
// 		goto parse;
// 	} else {
// 		// SFT_TCP_LOG(ERR, "missed TCP END option\n");
// 		ret = -1;
// 	}
// out:
	return 0;
}

__global__ void _packet_gather_kernel(
  doca_gpu_rxq_info*     rxq_info,
  doca_gpu_semaphore_in* sem_in,
  int32_t                sem_count,
  int32_t*               sem_idx,
  uint32_t*              timestamp_out,
  int64_t*               src_mac_out,
  int64_t*               dst_mac_out,
  int64_t*               src_ip_out,
  int64_t*               dst_ip_out,
  uint16_t*              src_port_out,
  uint16_t*              dst_port_out,
  int32_t*               data_offsets_out,
  char*                  data_out
)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ doca_gpu_semaphore_status sem_status;
	__shared__ uint32_t packet_count;
  __shared__ uintptr_t packet_address;

  if (threadIdx.x == 0) {

    doca_error_t ret;
    do
    {
      ret = doca_gpu_device_semaphore_get_value_status(
        sem_in + *sem_idx,
        DOCA_GPU_SEM_STATUS_HOLD,
        &sem_status,
        &packet_count,
        &packet_address);

    } while(ret == DOCA_ERROR_NOT_FOUND and sem_status != DOCA_GPU_SEM_STATUS_HOLD);
  }

  __syncthreads();

  __shared__ uint32_t stride_start_idx;

  if (threadIdx.x == 0) {
    stride_start_idx = doca_gpu_device_comm_buf_get_stride_idx(
      &(rxq_info->comm_buf),
      packet_address
    );
  }

  __syncthreads();

  int32_t data_offsets[PACKETS_PER_THREAD];

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
      data_offsets[i] = 0;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto header_length = static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));
    auto data_size = total_length - header_length;

    data_offsets[i] = data_size;

    // TCP timestamp option
    timestamp_out[packet_idx] = tcp_parse_timestamp(packet_l4);

    // mac address
    auto src_mac = packet_l2->s_addr.addr_bytes; // 6 bytes
    auto dst_mac = packet_l2->d_addr.addr_bytes; // 6 bytes

    src_mac_out[packet_idx] = mac_bytes_to_int64(src_mac);
    dst_mac_out[packet_idx] = mac_bytes_to_int64(dst_mac);

    // ip address
    auto src_address  = packet_l3->src_addr;
    auto dst_address  = packet_l3->dst_addr;

    auto src_address_rev = (src_address & 0x000000ff) << 24
                          | (src_address & 0x0000ff00) << 8
                          | (src_address & 0x00ff0000) >> 8
                          | (src_address & 0xff000000) >> 24;

    auto dst_address_rev = (dst_address & 0x000000ff) << 24
                          | (dst_address & 0x0000ff00) << 8
                          | (dst_address & 0x00ff0000) >> 8
                          | (dst_address & 0xff000000) >> 24;

    src_ip_out[packet_idx] = src_address_rev;
    dst_ip_out[packet_idx] = dst_address_rev;

    // ports
    auto src_port     = BYTE_SWAP16(packet_l4->src_port);
    auto dst_port     = BYTE_SWAP16(packet_l4->dst_port);

    src_port_out[packet_idx] = src_port;
    dst_port_out[packet_idx] = dst_port;
  }

  __syncthreads();

  BlockScan(temp_storage).ExclusiveSum(data_offsets, data_offsets);

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    uint8_t *packet = doca_gpu_device_comm_buf_get_stride_addr(
      &(rxq_info->comm_buf),
      stride_start_idx + packet_idx
    );

    rte_ether_hdr* packet_l2;
    rte_ipv4_hdr*  packet_l3;
    rte_tcp_hdr*   packet_l4;
    uint8_t*       packet_data;

    get_packet_tcp_headers(
      packet,
      &packet_l2,
      &packet_l3,
      &packet_l4,
      &packet_data
    );

    auto total_length = static_cast<int32_t>(BYTE_SWAP16(packet_l3->total_length));
    auto data_size = total_length - static_cast<int32_t>(packet_l4->dt_off * sizeof(int32_t));

    // printf("packet_idx(%d) data_offset(%d)\n", packet_idx, data_offsets[i]);

    data_offsets_out[packet_idx] = data_offsets[i];

    for (auto data_idx = 0; data_idx < data_size; data_idx++)
    {
      data_out[data_offsets[i] + data_idx] = packet_data[data_idx];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    doca_gpu_device_semaphore_update_status(
      sem_in + *sem_idx,
      DOCA_GPU_SEM_STATUS_FREE
    );
  }
}

namespace morpheus {
namespace doca {

namespace {

struct integers_to_mac_fn {
  cudf::column_device_view const d_column;
  int32_t const* d_offsets;
  char* d_chars;

  __device__ void operator()(cudf::size_type idx)
  {
    int64_t mac_address = d_column.element<int64_t>(idx);
    char* out_ptr       = d_chars + d_offsets[idx];
    
    mac_int64_to_chars(mac_address, out_ptr);
  }
};

}

std::unique_ptr<cudf::column> integers_to_mac(
  cudf::column_view const& integers,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr
)
{
  CUDF_EXPECTS(integers.type().id() == cudf::type_id::INT64, "Input column must be type_id::INT64 type");
  CUDF_EXPECTS(integers.null_count() == 0, "integers_to_mac does not support null values.");

  cudf::size_type strings_count = integers.size();

  if (strings_count == 0)
  {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  auto offsets_transformer_itr = thrust::constant_iterator<int32_t>(17);
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr,
    offsets_transformer_itr + strings_count,
    stream,
    mr
  );

  auto d_offsets = offsets_column->view().data<int32_t>();

  auto column   = cudf::column_device_view::create(integers, stream);
  auto d_column = *column;

  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);

  auto chars_column = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    integers_to_mac_fn{d_column, d_offsets, d_chars}
  );

  return cudf::make_strings_column(strings_count,
    std::move(offsets_column),
    std::move(chars_column),
    0,
    {});
}

struct picker {
  uint32_t* lengths;
  __device__ uint32_t operator()(cudf::size_type idx){
    if (lengths[idx] > 0)
    {
      printf("pdl: %d\n", lengths[idx]);
    }
    return lengths[idx];
  }
};

void packet_receive_kernel(
  doca_gpu_rxq_info*     rxq_info,
  doca_gpu_semaphore_in* sem_in,
  int32_t                sem_count,
  int32_t*               sem_idx,
  int32_t*               packet_count,
  int32_t*               packet_data_size,
  cudaStream_t           stream
)
{
  _packet_receive_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    packet_count,
    packet_data_size
  );
}

void packet_gather_kernel(
  doca_gpu_rxq_info*     rxq_info,
  doca_gpu_semaphore_in* sem_in,
  int32_t                sem_count,
  int32_t*               sem_idx,
  uint32_t*              timestamp_out,
  int64_t*               src_mac_out,
  int64_t*               dst_mac_out,
  int64_t*               src_ip_out,
  int64_t*               dst_ip_out,
  uint16_t*              src_port_out,
  uint16_t*              dst_port_out,
  int32_t*               data_offsets_out,
  char*                  data_out,
  cudaStream_t           stream
)
{
  _packet_gather_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    timestamp_out,
    src_mac_out,
    dst_mac_out,
    src_ip_out,
    dst_ip_out,
    src_port_out,
    dst_port_out,
    data_offsets_out,
    data_out
  );
}

}
}
