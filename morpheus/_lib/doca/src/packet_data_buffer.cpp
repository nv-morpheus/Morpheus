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

packet_data_buffer::packet_data_buffer(
    std::size_t buffer_size_bytes,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr):
    buffer{buffer_size_bytes, stream, mr},
    cur_offset_bytes{0},
    elements{0}
{
}

morpheus::TensorSize packet_data_buffer::capacity() const
{
    return buffer.size();
}

morpheus::TensorSize packet_data_buffer::available_bytes() const
{
    return capacity() - cur_offset_bytes;
}

bool packet_data_buffer::empty() const
{
    return cur_offset_bytes == 0;
}

void packet_data_buffer::advance(morpheus::TensorSize num_bytes, morpheus::TensorSize num_elements)
{
    cur_offset_bytes += num_bytes;
    elements += num_elements;
    CHECK(cur_offset_bytes <= capacity());
}

void packet_data_buffer::shrink_to_fit()
{
    if (available_bytes() > 0)
    {
        buffer.resize(cur_offset_bytes, buffer.stream());
        buffer.shrink_to_fit(buffer.stream());
    }
}

} // namespace morpheus::doca