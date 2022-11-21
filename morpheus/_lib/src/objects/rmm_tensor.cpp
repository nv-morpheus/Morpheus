/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/rmm_tensor.hpp"

#include "morpheus/objects/dev_mem_info.hpp"  // for DevMemInfo
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for get_element_stride

#include <cuda_runtime.h>            // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToDevice
#include <glog/logging.h>            // for DCHECK_LT, COMPACT_GOOGLE_LOG_FATAL, DCHECK, DCHECK_EQ, LogMessageFatal
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>
#include <srf/cuda/common.hpp>  // for SRF_CHECK_CUDA

#include <algorithm>  // for copy, transform
#include <cstdint>
#include <functional>  // for multiplies, plus, minus
#include <iterator>    // for back_insert_iterator, back_inserter
#include <memory>
#include <numeric>  // for accumulate, transform_reduce
#include <ostream>  // needed for logging
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** RMMTensor****************************************/
RMMTensor::RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
                     size_t offset,
                     DType dtype,
                     std::vector<TensorIndex> shape,
                     std::vector<TensorIndex> stride) :
  m_md(std::move(device_buffer)),
  m_offset(offset),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride))
{
    if (m_stride.empty())
    {
        detail::validate_stride(this->m_shape, this->m_stride);
    }

    DCHECK(m_offset + this->bytes() <= m_md->size())
        << "Inconsistent tensor. Tensor values would extend past the end of the device_buffer";
}

std::shared_ptr<MemoryDescriptor> RMMTensor::get_memory() const
{
    return nullptr;
}

void *RMMTensor::data() const
{
    return static_cast<uint8_t *>(m_md->data()) + this->offset_bytes();
}

RankType RMMTensor::rank() const
{
    return m_shape.size();
}

DType RMMTensor::dtype() const
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

void RMMTensor::get_shape(std::vector<TensorIndex> &s) const
{
    s.resize(rank());
    std::copy(m_shape.begin(), m_shape.end(), s.begin());
}

void RMMTensor::get_stride(std::vector<TensorIndex> &s) const
{
    s.resize(rank());
    std::copy(m_stride.begin(), m_stride.end(), s.begin());
}

bool RMMTensor::is_compact() const
{
    TensorIndex ttl = 1;
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

std::shared_ptr<ITensor> RMMTensor::slice(const std::vector<TensorIndex> &min_dims,
                                          const std::vector<TensorIndex> &max_dims) const
{
    // Calc new offset
    size_t offset = std::transform_reduce(
        m_stride.begin(), m_stride.end(), min_dims.begin(), m_offset, std::plus<>(), std::multiplies<>());

    // Calc new shape
    std::vector<TensorIndex> shape;
    std::transform(max_dims.begin(), max_dims.end(), min_dims.begin(), std::back_inserter(shape), std::minus<>());

    // Stride remains the same

    return std::make_shared<RMMTensor>(m_md, offset, m_dtype, shape, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::reshape(const std::vector<TensorIndex> &dims) const
{
    return std::make_shared<RMMTensor>(m_md, 0, m_dtype, dims, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::deep_copy() const
{
    // Deep copy
    std::shared_ptr<rmm::device_buffer> copied_buffer =
        std::make_shared<rmm::device_buffer>(*m_md, m_md->stream(), m_md->memory_resource());

    return std::make_shared<RMMTensor>(copied_buffer, m_offset, m_dtype, m_shape, m_stride);
}

std::shared_ptr<ITensor> RMMTensor::as_type(DType dtype) const
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

std::shared_ptr<ITensor> RMMTensor::copy_rows(const std::vector<std::pair<TensorIndex, TensorIndex>> &selected_rows,
                                              TensorIndex num_rows) const
{
    const auto tensor_type = dtype();
    const auto item_size   = tensor_type.item_size();
    const auto num_columns = static_cast<TensorIndex>(shape(1));
    const auto stride      = TensorUtils::get_element_stride<TensorIndex>(m_stride);
    const auto row_stride  = stride[0];

    auto output_buffer =
        std::make_shared<rmm::device_buffer>(num_rows * num_columns * item_size, rmm::cuda_stream_per_thread);

    auto output_offset = static_cast<uint8_t *>(output_buffer->data());

    for (const auto &rows : selected_rows)
    {
        const auto &sliced_input_tensor  = slice({rows.first, 0}, {rows.second, num_columns});
        const std::size_t num_input_rows = rows.second - rows.first;
        DCHECK_EQ(num_input_rows, sliced_input_tensor->shape(0));

        const auto slice_size = sliced_input_tensor->bytes();

        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            SRF_CHECK_CUDA(
                cudaMemcpy(output_offset, sliced_input_tensor->data(), slice_size, cudaMemcpyDeviceToDevice));
        }
        else
        {
            SRF_CHECK_CUDA(cudaMemcpy2D(output_offset,
                                        item_size,
                                        sliced_input_tensor->data(),
                                        row_stride * item_size,
                                        item_size,
                                        num_input_rows,
                                        cudaMemcpyDeviceToDevice));
        }

        output_offset += slice_size;
    }

    std::vector<TensorIndex> output_shape{num_rows, num_columns};
    return std::make_shared<RMMTensor>(output_buffer, 0, tensor_type, output_shape);
}
}  // namespace morpheus
