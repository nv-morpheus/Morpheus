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
#include <doca_eth_rxq.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_rxq.cuh>
#include <doca_gpunetio_dev_sem.cuh>
#include <matx.h>
#include <rmm/device_buffer.hpp>
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
  uint8_t*  payload_chars_out,
  int32_t* dst_offsets
)
{
  int pkt_idx = threadIdx.x;
  int j = 0;

  while (pkt_idx < packet_count) {
    uint8_t* pkt_hdr_addr = (uint8_t*)(packets_buffer[pkt_idx] + header_sizes[pkt_idx]);
    const int32_t dst_offset = dst_offsets[pkt_idx];
    const uint32_t payload_size = payload_sizes[pkt_idx];

    for (j = 0; j < payload_size; ++j) {
      payload_chars_out[dst_offset + j] = pkt_hdr_addr[j];
    }

    pkt_idx += blockDim.x;
  }

}

__global__ void _packet_gather_src_ip_kernel(
  int32_t   packet_count,
  uintptr_t*  packets_buffer,
  uint32_t* header_sizes,
  uint32_t* payload_sizes,
  uint32_t*  dst_buff
)
{
  int pkt_idx = threadIdx.x;

  while (pkt_idx < packet_count) {
    uint8_t* pkt_hdr_addr = (uint8_t*)(packets_buffer[pkt_idx]);
    dst_buff[pkt_idx] = ((struct eth_ip *)pkt_hdr_addr)->l3_hdr.src_addr;
    pkt_idx += blockDim.x;
  }
}

namespace morpheus {
namespace doca {

uint32_t gather_sizes(
    int32_t packet_count,
    uint32_t* size_list,
    rmm::cuda_stream_view stream
)
{
    auto sizes_tensor = matx::make_tensor<uint32_t>(size_list, {packet_count});
    auto bytes_tensor = matx::make_tensor<uint32_t>({1});

    (bytes_tensor = matx::sum(sizes_tensor)).run(stream.value());

    cudaStreamSynchronize(stream);
    return bytes_tensor(0);
}

rmm::device_buffer sizes_to_offsets(
    int32_t packet_count,
    uint32_t* sizes_buff,
    rmm::cuda_stream_view stream)
{
    // The cudf offsets column wants int32
    const auto out_elem_count = packet_count+1;
    const auto out_byte_size = out_elem_count*sizeof(int32_t);
    rmm::device_buffer out_buffer(out_byte_size, stream);

    auto sizes_tensor = matx::make_tensor<uint32_t>(sizes_buff, {packet_count});
    auto cum_tensor = matx::make_tensor<int32_t>({packet_count});

    // first element needs to be a 0
    auto zero_tensor = matx::make_tensor<int32_t>({1});
    zero_tensor.SetVals({0});

    auto offsets_tensor = matx::make_tensor<int32_t>(static_cast<int32_t*>(out_buffer.data()), {out_elem_count});


    (cum_tensor = matx::cumsum(matx::as_type<int32_t>(sizes_tensor))).run(stream.value());
    (offsets_tensor = matx::concat(0, zero_tensor, cum_tensor)).run(stream.value());

    cudaStreamSynchronize(stream);

    return out_buffer;
}

rmm::device_buffer sizes_to_offsets(
    int32_t packet_count,
    uint32_t* header_sizes_buff,
    uint32_t* payload_sizes_buff,
    rmm::cuda_stream_view stream)
{
    std::cerr << "sizes_to_offsets\n";
    const auto out_elem_count = packet_count+1;
    const auto out_byte_size = out_elem_count*sizeof(int32_t);
    rmm::device_buffer out_buffer(out_byte_size, stream);

    auto header_sizes_tensor = matx::make_tensor<uint32_t>(header_sizes_buff, {packet_count});
    auto payload_sizes_tensor = matx::make_tensor<uint32_t>(payload_sizes_buff, {packet_count});
    auto sizes_tensor = matx::make_tensor<int32_t>({packet_count});
    auto cum_tensor = matx::make_tensor<int32_t>({packet_count});

    // first element needs to be a 0
    auto zero_tensor = matx::make_tensor<int32_t>({1});
    zero_tensor.SetVals({0});

    auto offsets_tensor = matx::make_tensor<int32_t>(static_cast<int32_t*>(out_buffer.data()), {out_elem_count});


    (sizes_tensor = matx::as_type<int32_t>(header_sizes_tensor) + matx::as_type<int32_t>(payload_sizes_tensor)).run(stream.value());
    (cum_tensor = matx::cumsum(sizes_tensor)).run(stream.value());
    (offsets_tensor = matx::concat(0, zero_tensor, cum_tensor)).run(stream.value());

    std::cerr << "sizes_to_offsets - done\n";
    cudaStreamSynchronize(stream);
    std::cerr << "sizes_to_offsets - synced\n";

    return out_buffer;
}

void gather_header(
  int32_t      packet_count,
  uintptr_t*   packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  uint32_t*     dst_buff,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  _packet_gather_src_ip_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    dst_buff
  );

}

void gather_payload(
  int32_t      packet_count,
  uintptr_t*   packets_buffer,
  uint32_t*    header_sizes,
  uint32_t*    payload_sizes,
  uint8_t*     dst_buff,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto dst_offsets = sizes_to_offsets(packet_count, payload_sizes, stream);
  _packet_gather_payload_kernel<<<1, THREADS_PER_BLOCK, 0, stream>>>(
    packet_count,
    packets_buffer,
    header_sizes,
    payload_sizes,
    dst_buff,
    static_cast<int32_t*>(dst_offsets.data())
  );
}

} //doca
} //morpheus
