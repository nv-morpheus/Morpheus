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

#include "morpheus/types.hpp"  // for TensorSize

#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace morpheus::doca {

struct packet_data_buffer
{
    packet_data_buffer(std::size_t buffer_size_bytes,
                       rmm::cuda_stream_view stream,
                       rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

    morpheus::TensorSize capacity() const;

    morpheus::TensorSize available_bytes() const;

    bool empty() const;

    void advance(morpheus::TensorSize num_bytes, morpheus::TensorSize num_elements);

    template <typename T = uint8_t>
    T* data()
    {
        return static_cast<T*>(buffer.data());
    }

    template <typename T = uint8_t>
    T* current_location()
    {
        // Get the head in bytes to perform the offset math, then cast to the user's desired type
        return reinterpret_cast<T*>(data<uint8_t>() + cur_offset_bytes);
    }

    void shrink_to_fit();

    rmm::device_buffer buffer;
    morpheus::TensorSize cur_offset_bytes;
    morpheus::TensorSize elements;
};
}  // namespace morpheus::doca
