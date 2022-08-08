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

#include "morpheus/messages/memory/tensor_memory.hpp"

#include "morpheus/objects/tensor.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/
TensorMemory::TensorMemory(size_t count) : count(count) {}
TensorMemory::TensorMemory(size_t count, tensor_map_t &&tensors) : count(count), tensors(std::move(tensors)) {}

bool TensorMemory::has_tensor(const std::string &name) const
{
    return this->tensors.find(name) != this->tensors.end();
}

TensorMemory::tensor_map_t TensorMemory::copy_tensor_ranges(
    const std::vector<std::pair<TensorIndex, TensorIndex>> &ranges, size_t num_selected_rows) const
{
    tensor_map_t tensors;
    for (const auto &p : this->tensors)
    {
        tensors.insert(std::pair{p.first, p.second.copy_rows(ranges, num_selected_rows)});
    }

    return tensors;
}

}  // namespace morpheus
