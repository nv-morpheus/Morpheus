#include "morpheus/utilities/error.hpp"
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_ether.h>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <doca_gpunetio_dev_buf.cuh>
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

#define ETHER_ADDR_LEN  6 /**< Length of Ethernet address. */

#define BYTE_SWAP16(v) \
	((((uint16_t)(v) & UINT16_C(0x00ff)) << 8) | (((uint16_t)(v) & UINT16_C(0xff00)) >> 8))

#define TCP_PROTOCOL_ID 0x6
#define UDP_PROTOCOL_ID 0x11

enum tcp_flags {
	TCP_FLAG_FIN = (1 << 0),
	/* set tcp packet with Fin flag */
	TCP_FLAG_SYN = (1 << 1),
	/* set tcp packet with Syn flag */
	TCP_FLAG_RST = (1 << 2),
	/* set tcp packet with Rst flag */
	TCP_FLAG_PSH = (1 << 3),
	/* set tcp packet with Psh flag */
	TCP_FLAG_ACK = (1 << 4),
	/* set tcp packet with Ack flag */
	TCP_FLAG_URG = (1 << 5),
	/* set tcp packet with Urg flag */
	TCP_FLAG_ECE = (1 << 6),
	/* set tcp packet with ECE flag */
	TCP_FLAG_CWR = (1 << 7),
	/* set tcp packet with CQE flag */
};

struct ether_hdr {
	uint8_t d_addr_bytes[ETHER_ADDR_LEN];	/* Destination addr bytes in tx order */
	uint8_t s_addr_bytes[ETHER_ADDR_LEN];	/* Source addr bytes in tx order */
	uint16_t ether_type;			/* Frame type */
} __attribute__((__packed__));

struct ipv4_hdr {
	uint8_t version_ihl;		/* version and header length */
	uint8_t  type_of_service;	/* type of service */
	uint16_t total_length;		/* length of packet */
	uint16_t packet_id;		/* packet ID */
	uint16_t fragment_offset;	/* fragmentation offset */
	uint8_t  time_to_live;		/* time to live */
	uint8_t  next_proto_id;		/* protocol ID */
	uint16_t hdr_checksum;		/* header checksum */
	uint32_t src_addr;		/* source address */
	uint32_t dst_addr;		/* destination address */
} __attribute__((__packed__));

struct tcp_hdr {
	uint16_t src_port;	/* TCP source port */
	uint16_t dst_port;	/* TCP destination port */
	uint32_t sent_seq;	/* TX data sequence number */
	uint32_t recv_ack;	/* RX data acknowledgment sequence number */
	uint8_t dt_off;		/* Data offset */
	uint8_t tcp_flags;	/* TCP flags */
	uint16_t rx_win;	/* RX flow control window */
	uint16_t cksum;		/* TCP checksum */
	uint16_t tcp_urp;	/* TCP urgent pointer, if any */
} __attribute__((__packed__));

struct eth_ip_tcp_hdr {
	struct ether_hdr l2_hdr;	/* Ethernet header */
	struct ipv4_hdr l3_hdr;		/* IP header */
	struct tcp_hdr l4_hdr;		/* TCP header */
} __attribute__((__packed__));

__device__ __inline__ int
raw_to_tcp(const uintptr_t buf_addr, struct eth_ip_tcp_hdr **hdr, uint8_t **packet_data)
{
	(*hdr) = (struct eth_ip_tcp_hdr *) buf_addr;
	(*packet_data) = (uint8_t *) (buf_addr + sizeof(struct ether_hdr) + sizeof(struct ipv4_hdr) + (((*hdr)->l4_hdr.dt_off >> 4) * sizeof(int)));

	return 0;
}

__device__ __forceinline__ uint8_t
gpu_ipv4_hdr_len(const struct ipv4_hdr& hdr)
{
	return (uint8_t)((hdr.version_ihl & RTE_IPV4_HDR_IHL_MASK) * RTE_IPV4_IHL_MULTIPLIER);
};

__device__ __forceinline__ int32_t
get_payload_size(ipv4_hdr& packet_l3, tcp_hdr& packet_l4)
{
  auto total_length      = static_cast<int32_t>(BYTE_SWAP16(packet_l3.total_length));
  auto ip_header_length  = gpu_ipv4_hdr_len(packet_l3);
  auto tcp_header_length = static_cast<int32_t>(packet_l4.dt_off * sizeof(int32_t));
  auto data_size         = total_length - ip_header_length - tcp_header_length;

  return data_size;
}

__device__ __forceinline__ uint32_t
get_packet_size(ipv4_hdr& packet_l3)
{
  return static_cast<int32_t>(BYTE_SWAP16(packet_l3.total_length));
}

__device__ __forceinline__ bool
is_tcp_packet(ipv4_hdr& packet_l3)
{
  return packet_l3.next_proto_id == IPPROTO_TCP;
}

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
uint32_t const PACKET_RX_TIMEOUT_NS = 5000000; // 5ms
// uint32_t const PACKET_RX_TIMEOUT_NS = 50000000; // 50ms

// what if I had receive, gather, and release kernels?
// what if I abstracted away the GPUNetIO aspects by making these calls templated?

__device__ uint32_t tcp_parse_timestamp(rte_tcp_hdr const *tcp)
{
	const uint8_t *tcp_opt = (typeof(tcp_opt))tcp + RTE_TCP_MIN_HDR_LEN;
	const uint8_t *tcp_data = (typeof(tcp_data))tcp + static_cast<int32_t>(tcp->dt_off * sizeof(int32_t));

  while (tcp_opt < tcp_data) {
    switch(tcp_opt[0]) {
      case RTE_TCP_OPT_END:
        return 0;
      case RTE_TCP_OPT_NOP:
        tcp_opt++;
        continue;
      case RTE_TCP_OPT_TIMESTAMP:
        return (static_cast<uint32_t>(tcp_opt[2]) << 24)
            | (static_cast<uint32_t>(tcp_opt[3]) << 16)
            | (static_cast<uint32_t>(tcp_opt[4]) << 8)
            | (static_cast<uint32_t>(tcp_opt[5]) << 0);
      default:
        if (tcp_opt[1] == 0) {
          return 0;
        } else {
          tcp_opt += tcp_opt[1];
        }
        continue;
    }
  }

	return 0;
}

__global__ void _packet_receive_kernel(
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_count_out,
  int32_t*                packet_size_total_out,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer
)
{
  if (threadIdx.x == 0)
  {
    *packet_count_out = 0;
    *packet_size_total_out = 0;
  }

  __shared__ uint32_t packet_count;
  __shared__ doca_gpu_semaphore_status sem_status;

  uint64_t packet_offset;

  if (threadIdx.x == 0)
  {
    while (true)
    {
      auto ret = doca_gpu_dev_sem_get_status(sem_in, *sem_idx, &sem_status);

      if (ret != DOCA_SUCCESS) {
        // handle this eventually
      }

      if (sem_status == DOCA_GPU_SEMAPHORE_STATUS_FREE)
      {
        break;
      }
    }
  }

  __syncthreads();

  DOCA_GPUNETIO_VOLATILE(packet_count) = 0;

  __syncthreads();

  auto ret = doca_gpu_dev_eth_rxq_receive_block(
    rxq_info,
    PACKETS_PER_BLOCK,
    PACKET_RX_TIMEOUT_NS,
    &packet_count,
    &packet_offset
  );

  __threadfence();
  __syncthreads();

  if (packet_count == 0) {
    return;
  }

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    doca_gpu_buf *buf_ptr;
    doca_gpu_dev_eth_rxq_get_buf(rxq_info, packet_offset + packet_idx, &buf_ptr);

    uintptr_t buf_addr;
    doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);

    // rte_ether_hdr* packet_l2;
    // rte_ipv4_hdr*  packet_l3;
    // rte_tcp_hdr*   packet_l4;
    // uint8_t*       packet_data;

    // get_packet_tcp_headers(
    //   buf_addr,
    //   &packet_l2,
    //   &packet_l3,
    //   &packet_l4,
    //   &packet_data
    // );


    struct eth_ip_tcp_hdr *hdr;
    uint8_t *packet_data;
		raw_to_tcp(buf_addr, &hdr, &packet_data);

    auto packet_size = get_packet_size(hdr->l3_hdr);

    // for (auto i = 0; i < packet_size; i++) {
    //   packet_buffer[(packet_idx * 65536) + i] = packet_data[i];
    // }

    // raw_to_tcp(packet_buffer + (packet_idx * 65536), &hdr, &packet_data);
    // get_packet_tcp_headers(
    //   packet_buffer + (packet_idx * 65536),
    //   &packet_l2,
    //   &packet_l3,
    //   &packet_l4,
    //   &packet_data
    // );

    auto data_size = get_payload_size(hdr->l3_hdr, hdr->l4_hdr);

    packet_sizes[packet_idx] = data_size;

    if (is_tcp_packet(hdr->l3_hdr))
    {
      // printf("tid(%03i) pid(%04i) RK1 data_size(%i)\n", threadIdx.x, packet_idx, data_size);

      atomicAdd(packet_size_total_out, data_size);
      atomicAdd(packet_count_out, 1);
    }

    // printf("packet_idx(%d) data_size(%d) atom\n", packet_idx, data_size);
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    if (*packet_count_out > 0) {
      doca_gpu_dev_sem_set_packet_info(
        sem_in,
        *sem_idx,
        DOCA_GPU_SEMAPHORE_STATUS_HOLD,
        packet_count,
        packet_offset
      );
    } else {
      doca_gpu_dev_sem_set_status(
        sem_in,
        *sem_idx,
        DOCA_GPU_SEMAPHORE_STATUS_FREE
      );
    }
  }

  __threadfence();
  __syncthreads();
}

__global__ void _packet_gather_kernel(
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer,
  uint32_t*               timestamp_out,
  int64_t*                src_mac_out,
  int64_t*                dst_mac_out,
  int64_t*                src_ip_out,
  int64_t*                dst_ip_out,
  uint16_t*               src_port_out,
  uint16_t*               dst_port_out,
  int32_t*                data_offsets_out,
  int32_t*                data_size_out,
  int32_t*                tcp_flags_out,
  int32_t*                ether_type_out,
  int32_t*                next_proto_id_out,
  char*                   data_out,
  int32_t                 data_out_size
)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

	__shared__ uint32_t packet_count;

  uint64_t packet_offset;

  if (threadIdx.x == 0) {

    // printf("===== begin gk =====\n");

    doca_error_t ret;
    do
    {
      ret = doca_gpu_dev_sem_get_packet_info_status(
        sem_in,
        *sem_idx,
        DOCA_GPU_SEMAPHORE_STATUS_HOLD,
        &packet_count,
        &packet_offset);

    } while(ret == DOCA_ERROR_NOT_FOUND);

  }

  __syncthreads();

  int32_t data_capture[PACKETS_PER_THREAD];
  int32_t data_offsets[PACKETS_PER_THREAD];

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      data_capture[i] = 0;
      data_offsets[i] = 0;
      continue;
    }

    doca_gpu_buf *buf_ptr;
    doca_gpu_dev_eth_rxq_get_buf(rxq_info, packet_offset + packet_idx, &buf_ptr);

    uintptr_t buf_addr;
    doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);

    struct eth_ip_tcp_hdr *hdr;
    uint8_t *packet_data;
		raw_to_tcp(buf_addr, &hdr, &packet_data);

    auto data_size = get_payload_size(hdr->l3_hdr, hdr->l4_hdr);

    if (is_tcp_packet(hdr->l3_hdr))
    {
      data_capture[i] = 1;
      data_offsets[i] = data_size;

      // printf("tid(%03i) pid(%04i) GK1 data_size(%i)\n", threadIdx.x, packet_idx, data_size);
    }
    else
    {
      data_capture[i] = 0;
      data_offsets[i] = 0;
    }
  }

  __syncthreads();

  BlockScan(temp_storage).ExclusiveSum(data_offsets, data_offsets);

  __syncthreads();

  BlockScan(temp_storage).ExclusiveSum(data_capture, data_capture);

  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++)
  {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;

    if (packet_idx >= packet_count) {
      continue;
    }

    doca_gpu_buf *buf_ptr;
    doca_gpu_dev_eth_rxq_get_buf(rxq_info, packet_offset + packet_idx, &buf_ptr);

    uintptr_t buf_addr;
    doca_gpu_dev_buf_get_addr(buf_ptr, &buf_addr);

    struct eth_ip_tcp_hdr *hdr;
    uint8_t *packet_data;
		raw_to_tcp(buf_addr, &hdr, &packet_data);

    auto data_size = get_payload_size(hdr->l3_hdr, hdr->l4_hdr);

    if (not is_tcp_packet(hdr->l3_hdr))
    {
      continue;
    }

    // printf("tid(%03i) pid(%04i) GK2 data_size(%i)\n", threadIdx.x, packet_idx, data_size);

    auto packet_idx_out = data_capture[i];

    data_offsets_out[packet_idx_out] = data_offsets[i];

    for (auto data_idx = 0; data_idx < data_size; data_idx++)
    {
      auto value = packet_data[data_idx];

      auto data_out_idx = data_offsets[i] + data_idx;

      if (data_out_idx < data_out_size) {
        if((value >= 'a' and value <= 'z') or (value >= 'A' and value <= 'Z')) {
          data_out[data_out_idx] = value;
        } else {
          data_out[data_out_idx] = '_';
        }
      } else {
        // printf("tid(%03i) pid(%04i) OOB write(%i/%i): %c.\n", threadIdx.x, packet_idx, data_out_idx, data_out_size, value);
      }
    }

    // TCP timestamp option
    //timestamp_out[packet_idx_out] = tcp_parse_timestamp(packet_l4);

    auto now = cuda::std::chrono::system_clock::now();
    auto now_ms = cuda::std::chrono::time_point_cast<cuda::std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    timestamp_out[packet_idx_out] = epoch.count();

    // mac address
    auto src_mac = hdr->l2_hdr.s_addr_bytes; // 6 bytes
    auto dst_mac = hdr->l2_hdr.d_addr_bytes; // 6 bytes

    src_mac_out[packet_idx_out] = mac_bytes_to_int64(src_mac);
    dst_mac_out[packet_idx_out] = mac_bytes_to_int64(dst_mac);

    // ip address
    auto src_address  = hdr->l3_hdr.src_addr;
    auto dst_address  = hdr->l3_hdr.dst_addr;

    auto src_address_rev = (src_address & 0x000000ff) << 24
                          | (src_address & 0x0000ff00) << 8
                          | (src_address & 0x00ff0000) >> 8
                          | (src_address & 0xff000000) >> 24;

    auto dst_address_rev = (dst_address & 0x000000ff) << 24
                          | (dst_address & 0x0000ff00) << 8
                          | (dst_address & 0x00ff0000) >> 8
                          | (dst_address & 0xff000000) >> 24;

    src_ip_out[packet_idx_out] = src_address_rev;
    dst_ip_out[packet_idx_out] = dst_address_rev;

    // ports
    auto src_port     = BYTE_SWAP16(hdr->l4_hdr.src_port);
    auto dst_port     = BYTE_SWAP16(hdr->l4_hdr.dst_port);

    src_port_out[packet_idx_out] = src_port;
    dst_port_out[packet_idx_out] = dst_port;

    // packet size
    auto packet_size = get_packet_size(hdr->l3_hdr);
    data_size_out[packet_idx_out] = packet_size;

    // tcp flags
    auto tcp_flags = hdr->l4_hdr.tcp_flags;
    tcp_flags_out[packet_idx_out] = static_cast<int32_t> (tcp_flags);

    // frame type
    auto ether_type = hdr->l2_hdr.ether_type;
    ether_type_out[packet_idx_out] = static_cast<int32_t> (ether_type);

    // protocol id
    auto next_proto_id = hdr->l3_hdr.next_proto_id;
    next_proto_id_out[packet_idx_out] = static_cast<int32_t> (next_proto_id);
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    doca_gpu_dev_sem_set_status(
      sem_in,
      *sem_idx,
      DOCA_GPU_SEMAPHORE_STATUS_FREE
    );

    // printf("===== end gk =====\n");
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

  CHECK_CUDA(stream);

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
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_count,
  int32_t*                packet_size_total,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer,
  cudaStream_t            stream
)
{
  _packet_receive_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    packet_count,
    packet_size_total,
    packet_sizes,
    packet_buffer
  );

  CHECK_CUDA(stream);
}

void packet_gather_kernel(
  doca_gpu_eth_rxq*       rxq_info,
  doca_gpu_semaphore_gpu* sem_in,
  int32_t                 sem_count,
  int32_t*                sem_idx,
  int32_t*                packet_sizes,
  uint8_t*                packet_buffer,
  uint32_t*               timestamp_out,
  int64_t*                src_mac_out,
  int64_t*                dst_mac_out,
  int64_t*                src_ip_out,
  int64_t*                dst_ip_out,
  uint16_t*               src_port_out,
  uint16_t*               dst_port_out,
  int32_t*                data_offsets_out,
  int32_t*                data_size_out,
  int32_t*                tcp_flags_out,
  int32_t*                ether_type_out,
  int32_t*                next_proto_id_out,
  char*                   data_out,
  int32_t                 data_out_size,
  cudaStream_t            stream
)
{
  _packet_gather_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    rxq_info,
    sem_in,
    sem_count,
    sem_idx,
    packet_sizes,
    packet_buffer,
    timestamp_out,
    src_mac_out,
    dst_mac_out,
    src_ip_out,
    dst_ip_out,
    src_port_out,
    dst_port_out,
    data_offsets_out,
    data_size_out,
    tcp_flags_out,
    ether_type_out,
    next_proto_id_out,
    data_out,
    data_out_size
  );

  CHECK_CUDA(stream);
}

}
}
