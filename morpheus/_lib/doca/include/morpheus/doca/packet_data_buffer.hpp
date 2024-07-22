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

#pragma once

#include <rmm/device_buffer.hpp>

#include <memory>


namespace morpheus::doca {

struct packet_data_buffer
{
    packet_data_buffer(std::size_t packet_count,
                       std::size_t header_size,
                       std::size_t payload_size,
                       std::size_t payload_sizes_size,    
                       rmm::cuda_stream_view rmm_stream,
                       rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

    std::size_t num_packets;
    rmm::cuda_stream_view stream;
    std::shared_ptr<rmm::device_buffer> header_buffer;
    std::unique_ptr<rmm::device_buffer> payload_buffer;
    std::unique_ptr<rmm::device_buffer> payload_sizes_buffer;
};
}  // namespace morpheus::doca
