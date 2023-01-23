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

#include "morpheus/objects/dev_mem_info.hpp"

#include <glog/logging.h>  // for DCHECK

#include <cstdint>     // for uint8_t
#include <functional>  // for multiplies
#include <numeric>     // for accumulate
#include <utility>     // for move

namespace morpheus {
// Component public implementations
// ************ DevMemInfo************************* //

DevMemInfo::DevMemInfo(std::shared_ptr<rmm::device_buffer> buffer,
                       DType dtype,
                       std::vector<std::size_t> shape,
                       std::vector<std::size_t> stride,
                       size_t offset_bytes) :
  m_buffer(std::move(buffer)),
  m_dtype(std::move(dtype)),
  m_shape(std::move(shape)),
  m_stride(std::move(stride)),
  m_offset_bytes(offset_bytes)
{
    DCHECK(m_offset_bytes + this->bytes() <= m_buffer->size())
        << "Inconsistent dimensions, values would extend past the end of the device_buffer";
}

std::size_t DevMemInfo::bytes() const
{
    return count() * m_dtype.item_size();
}

std::size_t DevMemInfo::count() const
{
    return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
}

std::size_t DevMemInfo::offset_bytes() const
{
    return m_offset_bytes;
}

const DType& DevMemInfo::dtype() const
{
    return m_dtype;
}

TypeId DevMemInfo::type_id() const
{
    return m_dtype.type_id();
}

const std::vector<std::size_t>& DevMemInfo::shape() const
{
    return m_shape;
}

std::size_t DevMemInfo::shape(std::size_t idx) const
{
    return m_shape.at(idx);
}

const std::vector<std::size_t>& DevMemInfo::stride() const
{
    return m_stride;
}

std::size_t DevMemInfo::stride(std::size_t idx) const
{
    return m_stride.at(idx);
}

std::unique_ptr<rmm::device_buffer> DevMemInfo::make_new_buffer(std::size_t bytes) const
{
    return std::make_unique<rmm::device_buffer>(bytes, m_buffer->stream(), m_buffer->memory_resource());
}

void* DevMemInfo::data() const
{
    return static_cast<uint8_t*>(m_buffer->data()) + m_offset_bytes;
}
}  // namespace morpheus
