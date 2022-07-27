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

#include "morpheus/objects/tensor.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/

/**
 * @brief Container for holding a collection of named `TensorObject`s in a `std::map` keyed by name.
 * Base class for `InferenceMemory` & `ResponseMemory`.
 *
 */
class TensorMemory
{
  public:
    using tensor_map_t = std::map<std::string, TensorObject>;

    TensorMemory(size_t count);
    TensorMemory(size_t count, tensor_map_t &&tensors);

    size_t count{0};
    tensor_map_t tensors;

    bool has_tensor(const std::string &name) const;
    tensor_map_t copy_tensor_ranges(const std::vector<std::pair<TensorIndex, TensorIndex>> &ranges,
                                    size_t num_selected_rows) const;
};

}  // namespace morpheus
