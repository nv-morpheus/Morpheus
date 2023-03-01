/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

/**
 * @brief Struct describing device memory resources.
 *
 */
struct MemoryDescriptor
{
    /**
     * @brief Construct a new MemoryDescriptor object.  If `memory_resource` is null the value returned by
     * `rmm::mr::get_current_device_resource()` will be used.
     *
     * @param stream
     * @param memory_resource
     */
    MemoryDescriptor(rmm::cuda_stream_view stream                  = rmm::cuda_stream_per_thread,
                     rmm::mr::device_memory_resource* mem_resource = nullptr);
    MemoryDescriptor(MemoryDescriptor& other)  = default;
    MemoryDescriptor(MemoryDescriptor&& other) = default;

    // Cuda stream
    rmm::cuda_stream_view cuda_stream;

    // Memory resource
    rmm::mr::device_memory_resource* memory_resource;
};
