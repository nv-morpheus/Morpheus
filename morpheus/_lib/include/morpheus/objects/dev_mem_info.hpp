/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/memory_descriptor.hpp"

#include <rmm/device_buffer.hpp>

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr, unique_ptr & make_unique
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** DevMemInfo******************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * @brief Simple object that describes a buffer in device memory
 */
class DevMemInfo
{
  public:
    /**
     * @brief Construct a new DevMemInfo object.
     *
     * @param data
     * @param dtype
     * @param shape
     * @param stride
     * @param offset_bytes
     * @param stream
     * @param memory_resource
     */
    DevMemInfo(void* data,
               DType dtype,
               std::shared_ptr<MemoryDescriptor> md,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> stride,
               size_t offset_bytes = 0);

    /**
     * @brief Construct a new DevMemInfo object from an existing `rmm::device_buffer`.
     *
     * @param buffer
     * @param dtype
     * @param shape
     * @param stride
     * @param offset_bytes
     */
    DevMemInfo(std::shared_ptr<rmm::device_buffer> buffer,
               DType dtype,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> stride,
               size_t offset_bytes = 0);
    DevMemInfo(DevMemInfo&& other) = default;

    /**
     * @brief Return the number of bytes stored in the underlying buffer
     *
     * @return std::size_t
     */
    std::size_t bytes() const;

    /**
     * @brief Return the element count stored in the underlying buffer
     *
     * @return std::size_t
     */
    std::size_t count() const;

    /**
     * @brief Return the number of bytes offset from the head of the buffer
     *
     * @return std::size_t
     */
    std::size_t offset_bytes() const;

    /**
     * @brief Return the type of the data stored in the buffer
     *
     * @return const DType&
     */
    const DType& dtype() const;

    /**
     * @brief Return the type id of the data stored in the buffer
     *
     * @return TypeId
     */
    TypeId type_id() const;

    /**
     * @brief Return a reference to the shape
     *
     * @return const std::vector<std::size_t>&
     */
    const std::vector<std::size_t>& shape() const;

    /**
     * @brief Return a the dimension at `idx`
     *
     * @param idx
     * @return std::size_t
     */
    std::size_t shape(std::size_t idx) const;

    /**
     * @brief Return a reference to the stride expressed in elements
     *
     * @return const std::vector<std::size_t>&
     */
    const std::vector<std::size_t>& stride() const;

    /**
     * @brief Return the stride at `idx`
     *
     * @param idx
     * @return std::size_t
     */
    std::size_t stride(std::size_t idx) const;

    /**
     * @brief Returns raw pointer to underlying buffer offset by the `offset`
     *
     * @return void*
     */
    void* data() const;

    /**
     * @brief Return the memory descriptor
     *
     * @return std::shared_ptr<MemoryDescriptor>
     */
    std::shared_ptr<MemoryDescriptor> memory() const;

    /**
     * @brief Constructs a new rmm buffer with the same stream and memory resource as the current buffer
     *
     * @param bytes
     * @return std::unique_ptr<rmm::device_buffer>
     */
    std::unique_ptr<rmm::device_buffer> make_new_buffer(std::size_t bytes) const;

  private:
    // Pointer to the head of our data
    void* m_data;

    // Type of elements in the buffer
    const DType m_dtype;

    // Shape & stride of the data in the buffer
    const std::vector<std::size_t> m_shape;
    const std::vector<std::size_t> m_stride;

    // Offset from head of data in bytes
    const size_t m_offset_bytes;

    // Device resources used to allocate this memory
    std::shared_ptr<MemoryDescriptor> m_md;
};

/** @} */  // end of group
}  // namespace morpheus
