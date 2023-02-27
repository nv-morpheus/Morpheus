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

#include "morpheus/objects/dtype.hpp"  // for DType
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/types.hpp"  // for RankType, shape_type_t, TensorIndex

#include <rmm/device_buffer.hpp>

#include <cstddef>  // for size_t
#include <memory>
#include <utility>  // for pair
#include <vector>

namespace morpheus {
#pragma GCC visibility push(default)
/****** Component public implementations *******************/
/****** RMMTensor****************************************/

/**
 * @addtogroup objects
 * @{
 * @file
 */

/**
 * TODO(Documentation)
 */
class RMMTensor : public ITensor
{
  public:
    RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
              size_t offset,
              DType dtype,
              shape_type_t shape,
              shape_type_t stride = {});

    ~RMMTensor() override = default;

    /**
     * TODO(Documentation)
     */
    bool is_compact() const final;

    /**
     * TODO(Documentation)
     */
    DType dtype() const override;

    /**
     * TODO(Documentation)
     */
    RankType rank() const final;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> deep_copy() const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> reshape(const shape_type_t& dims) const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> slice(const shape_type_t& min_dims, const shape_type_t& max_dims) const override;

    /**
     * @brief Creates a depp copy of the specified rows specified as vector<pair<start, stop>> not inclusive
     * of the stop row.
     *
     * @param selected_rows
     * @param num_rows
     * @return std::shared_ptr<ITensor>
     */
    std::shared_ptr<ITensor> copy_rows(const std::vector<std::pair<TensorIndex, TensorIndex>>& selected_rows,
                                       TensorIndex num_rows) const override;

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<MemoryDescriptor> get_memory() const override;

    /**
     * TODO(Documentation)
     */
    std::size_t bytes() const final;

    /**
     * TODO(Documentation)
     */
    std::size_t count() const final;

    /**
     * TODO(Documentation)
     */
    std::size_t shape(std::size_t idx) const final;

    /**
     * TODO(Documentation)
     */
    std::size_t stride(std::size_t idx) const final;

    /**
     * TODO(Documentation)
     */
    void* data() const override;

    /**
     * TODO(Documentation)
     */
    void get_shape(shape_type_t& s) const;

    /**
     * TODO(Documentation)
     */
    void get_stride(shape_type_t& s) const;

    // Tensor reshape(std::vector<TensorIndex> shape)
    // {
    //     CHECK(is_compact());
    //     return Tensor(descriptor_shared(), dtype_size(), shape);
    // }

    /**
     * TODO(Documentation)
     */
    std::shared_ptr<ITensor> as_type(DType dtype) const override;

  protected:
  private:
    /**
     * TODO(Documentation)
     */
    size_t offset_bytes() const;

    // Memory info
    std::shared_ptr<rmm::device_buffer> m_md;
    size_t m_offset;

    // // Type info
    // std::string m_typestr;
    // std::size_t m_dtype_size;
    DType m_dtype;

    // Shape info
    shape_type_t m_shape;
    shape_type_t m_stride;
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
