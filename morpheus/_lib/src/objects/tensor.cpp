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

#include "morpheus/objects/tensor.hpp"

#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils::get_element_stride
#include "morpheus/utilities/type_util.hpp"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyDeviceToHost
#include <rmm/device_buffer.hpp>
#include <srf/cuda/common.hpp>  // for SRF_CHECK_CUDA

#include <cstdint>
#include <memory>
#include <string>
#include <utility>  // for move
#include <vector>

namespace morpheus {
Tensor::Tensor(std::shared_ptr<rmm::device_buffer> buffer,
               std::string init_typestr,
               std::vector<int32_t> init_shape,
               std::vector<int32_t> init_strides,
               size_t init_offset) :
  m_device_buffer(std::move(buffer)),
  typestr(std::move(init_typestr)),
  shape(std::move(init_shape)),
  strides(std::move(init_strides)),
  m_offset(init_offset)
{}

void *Tensor::data() const
{
    return static_cast<uint8_t *>(m_device_buffer->data()) + m_offset;
}

size_t Tensor::bytes_count() const
{
    // temp just return without shape, size, offset, etc
    return m_device_buffer->size();
}

std::vector<uint8_t> Tensor::get_host_data() const
{
    std::vector<uint8_t> out_data;

    out_data.resize(this->bytes_count());

    SRF_CHECK_CUDA(cudaMemcpy(&out_data[0], this->data(), this->bytes_count(), cudaMemcpyDeviceToHost));

    return out_data;
}

auto Tensor::get_stream() const
{
    return this->m_device_buffer->stream();
}

TensorObject Tensor::create(std::shared_ptr<rmm::device_buffer> buffer,
                            DType dtype,
                            std::vector<TensorIndex> shape,
                            std::vector<TensorIndex> strides,
                            size_t offset)
{
    auto md = nullptr;

    if (!strides.empty())
    {
        strides = TensorUtils::get_element_stride<TensorIndex>(strides);
    }
    auto tensor = std::make_shared<RMMTensor>(buffer, offset, dtype, shape, strides);

    return TensorObject(md, tensor);
}
}  // namespace morpheus
