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

#include "morpheus/utilities/type_util.hpp"  // for DType
#include "morpheus/utilities/type_util_detail.hpp"

#include <rmm/device_buffer.hpp>

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr

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
    DevMemInfo(std::shared_ptr<rmm::device_buffer> buffer,
               DType dtype,
               std::vector<std::size_t> shape,
               std::vector<std::size_t> stride,
               size_t offset_bytes = 0);
    DevMemInfo(DevMemInfo&& other) = default;

    std::size_t bytes() const;
    std::size_t count() const;
    std::size_t offset_bytes() const;
    const DType& dtype() const;
    TypeId type_id() const;

    const std::vector<std::size_t>& shape() const;
    std::size_t shape(std::size_t idx) const;

    // Stride in elements
    const std::vector<std::size_t>& stride() const;
    std::size_t stride(std::size_t idx) const;

    /**
     * @brief Returns raw pointer to underlying buffer offset by the `offset`
     *
     * @return void*
     */
    void* data() const;
    rmm::cuda_stream_view stream () const;
    rmm::mr::device_memory_resource* memory_resource() const;

  private:
    // Buffer of data
    std::shared_ptr<rmm::device_buffer> m_buffer;

    // Type of elements in the buffer
    DType m_dtype;

    // Shape & stride of the data in the buffer
    std::vector<std::size_t> m_shape;
    std::vector<std::size_t> m_stride;

    // Offset from head of data in bytes
    size_t m_offset_bytes;
};

/** @} */  // end of group
}  // namespace morpheus
