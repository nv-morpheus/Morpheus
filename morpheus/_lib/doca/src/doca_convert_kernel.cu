/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "morpheus/doca/common.hpp"
#include "morpheus/doca/packets.hpp"
#include "morpheus/utilities/error.hpp"

#include <cub/cub.cuh>
#include <cuda/std/chrono>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <rmm/exec_policy.hpp>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <stdio.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <memory>

__global__ void _packet_gather_payload_kernel(
  int32_t  packet_count,
  uintptr_t*  packets_buffer,
  uint32_t* header_sizes,
  uint32_t* payload_sizes,
  uint8_t*  payload_chars_out
)
{
  int pkt_idx = threadIdx.x;
  int j = 0;

  while (pkt_idx < packet_count) {
    uint8_t* pkt_hdr_addr = (uint8_t*)(packets_buffer[pkt_idx] + header_sizes[pkt_idx]);
    for (j = 0; j < payload_sizes[pkt_idx]; j++)
      payload_chars_out[(MAX_PKT_SIZE * pkt_idx) + j] = pkt_hdr_addr[j];
    for (; j < MAX_PKT_SIZE; j++)
      payload_chars_out[(MAX_PKT_SIZE * pkt_idx) + j] = '\0';
    pkt_idx += blockDim.x;
  }

#if 0

  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;
  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;
  int32_t payload_offsets[PACKETS_PER_THREAD];
  /* Th0 will work on first 4 packets, etc.. */
  for (auto i = 0; i < PACKETS_PER_THREAD; i++) {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;
    if (packet_idx >= packet_count)
      payload_offsets[i] = 0;
    else
      payload_offsets[i] = payload_sizes[packet_idx];
  }
  __syncthreads();

  /* Calculate the right payload offset for each thread */
  int32_t data_offsets_agg;
  BlockScan(temp_storage).ExclusiveSum(payload_offsets, payload_offsets, data_offsets_agg);
  __syncthreads();

  for (auto i = 0; i < PACKETS_PER_THREAD; i++) {
    auto packet_idx = threadIdx.x * PACKETS_PER_THREAD + i;
    if (packet_idx >= packet_count)
      continue;

    auto payload_size = payload_sizes[packet_idx];
    for (auto j = 0; j < payload_size; j++) {
      auto value = *(((uint8_t*)packets_buffer[packet_idx]) + header_sizes[packet_idx] + j);
      payload_chars_out[payload_offsets[i] + j] = value;
      // printf("payload %d size %d : 0x%1x / 0x%1x addr %lx\n",
      //     payload_offsets[i] + j, payload_size,
      //     payload_chars_out[payload_offsets[i] + j], value,
      //     packets_buffer[packet_idx]);
    }
  }
#endif
}

__global__ void _packet_gather_header_kernel(
  int32_t   packet_count,
  uintptr_t*  packets_buffer,
  uint32_t* header_sizes,
  uint32_t* payload_sizes,
  uint8_t*  header_src_ip_addr
)
{
  int pkt_idx = threadIdx.x;

  while (pkt_idx < packet_count) {
    uint8_t* pkt_hdr_addr = (uint8_t*)(packets_buffer[pkt_idx]);
    int len = ip_to_string(((struct eth_ip *)pkt_hdr_addr)->l3_hdr.src_addr, header_src_ip_addr + (IP_ADDR_STRING_LEN * pkt_idx));
    while (len < IP_ADDR_STRING_LEN)
      header_src_ip_addr[(IP_ADDR_STRING_LEN * pkt_idx) + len++] = '\0';
    pkt_idx += blockDim.x;
  }
}

namespace morpheus {
namespace doca {

std::unique_ptr<cudf::column> gather_payload(
  int32_t      packet_count,
  uintptr_t*    packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  uint32_t*    fixed_size_list,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto [offsets_column, bytes] = cudf::detail::make_offsets_child_column(
    fixed_size_list,
    fixed_size_list + packet_count,
    stream,
    mr
  );

  auto chars_column = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<uint8_t>();

  _packet_gather_payload_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    d_chars
  );

  return cudf::make_strings_column(packet_count,
    std::move(offsets_column),
    std::move(chars_column),
    0,
    {});
}

std::unique_ptr<cudf::column> gather_header(
  int32_t      packet_count,
  uintptr_t*    packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto [offsets_column, bytes] = cudf::detail::make_offsets_child_column(
    header_sizes,
    header_sizes + packet_count,
    stream,
    mr
  );

  auto chars_column = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  uint8_t *d_chars      = chars_column->mutable_view().data<uint8_t>();

  _packet_gather_header_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    d_chars
  );

  return cudf::make_strings_column(packet_count,
    std::move(offsets_column),
    std::move(chars_column),
    0,
    {});
}

void gather_header_scalar(
  int32_t      packet_count,
  uintptr_t*    packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  uint8_t*      header_src_ip_addr,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
   _packet_gather_header_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    header_src_ip_addr
  );
}

void gather_payload_scalar(
  int32_t      packet_count,
  uintptr_t*    packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  uint8_t*      payload_col,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  _packet_gather_payload_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    payload_col
  );
}


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

  auto const_17_itr = thrust::constant_iterator<cudf::size_type>(17);
  auto [offsets_column, bytes] = cudf::detail::make_offsets_child_column(
    const_17_itr,
    const_17_itr + strings_count,
    stream,
    mr
  );

  auto column       = cudf::column_device_view::create(integers, stream);
  auto d_column     = *column;
  auto d_offsets    = offsets_column->view().data<int32_t>();
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


} //doca
} //morpheus
