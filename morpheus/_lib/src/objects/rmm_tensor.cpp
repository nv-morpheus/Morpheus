/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/objects/rmm_tensor.hpp>

#include <morpheus/utilities/matx_util.hpp>
#include <morpheus/utilities/type_util.hpp>

#include <morpheus/objects/tensor_object.hpp>

#include <pybind11/pybind11.h>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** RMMTensor****************************************/
RMMTensor::RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
                     size_t offset,
                     DType dtype,
                     std::vector<neo::TensorIndex> shape,
                     std::vector<neo::TensorIndex> stride) :
  m_md(std::move(device_buffer)),
  m_offset(offset),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride))
{
    if (m_stride.empty())
    {
        neo::detail::validate_stride(this->m_shape, this->m_stride);
    }

    DCHECK(m_offset + this->bytes() <= m_md->size())
        << "Inconsistent tensor. Tensor values would extend past the end of the device_buffer";
}

std::shared_ptr<neo::MemoryDescriptor> RMMTensor::get_memory() const
{
    return nullptr;
}

void *RMMTensor::data() const
{
    return static_cast<uint8_t *>(m_md->data()) + this->offset_bytes();
}

neo::RankType RMMTensor::rank() const
{
    return m_shape.size();
}

neo::DataType RMMTensor::dtype() const
{
    return m_dtype;
}

std::size_t RMMTensor::count() const
{
    return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
}

std::size_t RMMTensor::bytes() const
{
    return count() * m_dtype.item_size();
}

std::size_t RMMTensor::shape(std::size_t idx) const
{
    DCHECK_LT(idx, m_shape.size());
    return m_shape.at(idx);
}

std::size_t RMMTensor::stride(std::size_t idx) const
{
    DCHECK_LT(idx, m_stride.size());
    return m_stride.at(idx);
}

void RMMTensor::get_shape(std::vector<neo::TensorIndex> &s) const
{
    s.resize(rank());
    std::copy(m_shape.begin(), m_shape.end(), s.begin());
}

void RMMTensor::get_stride(std::vector<neo::TensorIndex> &s) const
{
    s.resize(rank());
    std::copy(m_stride.begin(), m_stride.end(), s.begin());
}

bool RMMTensor::is_compact() const
{
    neo::TensorIndex ttl = 1;
    for (int i = rank() - 1; i >= 0; i--)
    {
        if (stride(i) != ttl)
        {
            return false;
        }

        ttl *= shape(i);
    }
    return true;
}

std::shared_ptr<neo::ITensor> RMMTensor::slice(const std::vector<neo::TensorIndex> &min_dims,
                                               const std::vector<neo::TensorIndex> &max_dims) const
{
    // Calc new offset
    size_t offset = std::transform_reduce(
        m_stride.begin(), m_stride.end(), min_dims.begin(), m_offset, std::plus<>(), std::multiplies<>());

    // Calc new shape
    std::vector<neo::TensorIndex> shape;
    std::transform(max_dims.begin(), max_dims.end(), min_dims.begin(), std::back_inserter(shape), std::minus<>());

    // Stride remains the same

    return std::make_shared<RMMTensor>(m_md, offset, m_dtype, shape, m_stride);
}

std::shared_ptr<neo::ITensor> RMMTensor::reshape(const std::vector<neo::TensorIndex> &dims) const
{
    return std::make_shared<RMMTensor>(m_md, 0, m_dtype, dims, m_stride);
}

std::shared_ptr<neo::ITensor> RMMTensor::deep_copy() const
{
    // Deep copy
    std::shared_ptr<rmm::device_buffer> copied_buffer =
        std::make_shared<rmm::device_buffer>(*m_md, m_md->stream(), m_md->memory_resource());

    return std::make_shared<RMMTensor>(copied_buffer, m_offset, m_dtype, m_shape, m_stride);
}

std::shared_ptr<neo::ITensor> RMMTensor::as_type(neo::DataType dtype) const
{
    DType new_dtype(dtype.type_id());

    auto input_type  = m_dtype.type_id();
    auto output_type = new_dtype.type_id();

    // Now do the conversion
    auto new_data_buffer =
        MatxUtil::cast(DevMemInfo{this->count(), input_type, m_md, this->offset_bytes()}, output_type);

    // Return the new type
    return std::make_shared<RMMTensor>(new_data_buffer, 0, new_dtype, m_shape, m_stride);
}

size_t RMMTensor::offset_bytes() const
{
    return m_offset * m_dtype.item_size();
}
}  // namespace morpheus
