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

#include "morpheus/objects/dtype.hpp"  // for DType, TypeId
#include "morpheus/types.hpp"          // for ShapeType, TensorIndex

#include <rmm/device_buffer.hpp>

#include <memory>  // for shared_ptr, unique_ptr & make_unique
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
    DevMemInfo(std::shared_ptr<rmm::device_buffer> buffer,
               DType dtype,
               ShapeType shape,
               ShapeType stride,
               TensorIndex offset_bytes = 0);
    DevMemInfo(DevMemInfo&& other) = default;

    TensorIndex bytes() const;
    TensorIndex count() const;
    TensorIndex offset_bytes() const;
    const DType& dtype() const;
    TypeId type_id() const;

    const ShapeType& shape() const;
    TensorIndex shape(TensorIndex idx) const;

    // Stride in elements
    const ShapeType& stride() const;
    TensorIndex stride(TensorIndex idx) const;

    /**
     * @brief Returns raw pointer to underlying buffer offset by the `offset`
     *
     * @return void*
     */
    void* data() const;

    /**
     * @brief Constructs a new rmm buffer with the same stream and memory resource as the current buffer
     *
     * @param bytes
     * @return std::unique_ptr<rmm::device_buffer>
     */
    std::unique_ptr<rmm::device_buffer> make_new_buffer(TensorIndex bytes) const;

  private:
    // Buffer of data
    std::shared_ptr<rmm::device_buffer> m_buffer;

    // Type of elements in the buffer
    const DType m_dtype;

    // Shape & stride of the data in the buffer
    const ShapeType m_shape;
    const ShapeType m_stride;

    // Offset from head of data in bytes
    const TensorIndex m_offset_bytes;
};

/** @} */  // end of group
}  // namespace morpheus
