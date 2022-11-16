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

#pragma once

#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/type_util.hpp"

#include <rmm/device_buffer.hpp>

#include <cstddef>  // for size_t
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** Tensor****************************************/

/**
 * @addtogroup objects
 * @{
 * @file
*/

/**
 * TODO(Documentation)
 */

#pragma GCC visibility push(default)
class Tensor
{
  public:
    Tensor(std::shared_ptr<rmm::device_buffer> buffer,
           std::string init_typestr,
           std::vector<int32_t> init_shape,
           std::vector<int32_t> init_strides,
           size_t init_offset = 0);

    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    std::string typestr;

    /**
     * TODO(Documentation)
     */
    void *data() const;

    /**
     * TODO(Documentation)
     */
    size_t bytes_count() const;

    /**
     * TODO(Documentation)
     */
    std::vector<uint8_t> get_host_data() const;

    /**
     * TODO(Documentation)
     */
    auto get_stream() const;

    /**
     * TODO(Documentation)
     */
    static TensorObject create(std::shared_ptr<rmm::device_buffer> buffer,
                               DType dtype,
                               std::vector<TensorIndex> shape,
                               std::vector<TensorIndex> strides,
                               size_t offset = 0);

  private:
    size_t m_offset;
    std::shared_ptr<rmm::device_buffer> m_device_buffer;
};

#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
