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

#include "morpheus/objects/tensor_object.hpp"

#include "morpheus/utilities/tensor_util.hpp"

#include <srf/memory/blob.hpp>       // for blob

#include <vector>

namespace morpheus {

static void set_contiguous_stride(const std::vector<TensorIndex>& shape, std::vector<TensorIndex>& stride)
{
    stride.resize(shape.size());
    TensorIndex ttl = 1;
    auto rank       = shape.size();
    for (int i = rank - 1; i >= 0; i--)
    {
        stride[i] = ttl;
        ttl *= shape.at(i);
    }
}

TensorView::TensorView(srf::memory::blob bv, DataType dtype, std::vector<TensorIndex> shape) :
  srf::memory::blob(std::move(bv)),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape))
{
    TensorUtils::set_contiguous_stride(m_shape, m_stride);

    // validate the memory block defined by the blob has sufficient capacity to
    // hold a tensor of shape/stride.
}

TensorView::TensorView(srf::memory::blob bv,
                       DataType dtype,
                       std::vector<TensorIndex> shape,
                       std::vector<TensorIndex> stride) :
  srf::memory::blob(std::move(bv)),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride))
{
    CHECK_EQ(m_shape.size(), m_stride.size());

    // for now, we are only supporting row-major in the TensorView

    // validate row-major

    // validate the memory block defined by the blob has sufficient capacity to
    // hold a tensor of shape/stride.
}

bool TensorView::is_contiguous() const
{
    return TensorUtils::has_contiguous_stride(shape(), stride());
}

const DataType& TensorView::dtype() const
{
    return m_dtype;
}
const std::vector<TensorIndex>& TensorView::shape() const
{
    return m_shape;
}
const std::vector<TensorIndex>& TensorView::stride() const
{
    return m_stride;
}
}  // namespace morpheus
