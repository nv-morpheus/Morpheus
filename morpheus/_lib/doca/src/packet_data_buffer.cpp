/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/doca/packet_data_buffer.hpp"

#include <glog/logging.h>

namespace morpheus::doca {

packet_data_buffer::packet_data_buffer() :
  m_num_packets{0},
  m_stream{rmm::cuda_stream_per_thread},
  m_header_buffer{nullptr},
  m_payload_buffer{nullptr},
  m_payload_offsets_buffer{nullptr}
{}

packet_data_buffer::packet_data_buffer(std::size_t num_packets,
                                       std::size_t header_size,
                                       std::size_t payload_size,
                                       std::size_t payload_sizes_size, 
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr) :
  m_num_packets{num_packets},
  m_stream{stream},
  m_header_buffer{std::make_shared<rmm::device_buffer>(header_size, stream, mr)},
  m_payload_buffer{std::make_unique<rmm::device_buffer>(payload_size, stream, mr)},
  m_payload_offsets_buffer{std::make_unique<rmm::device_buffer>(payload_sizes_size, stream, mr)}
{}

packet_data_buffer packet_data_buffer::combine_buffers(std::vector<doca::packet_data_buffer>&& packets)
{
  DCHECK(!packets.empty());

  if (packets.size() == 1)
  {
    return packets[0];
  }
}

}  // namespace morpheus::doca
