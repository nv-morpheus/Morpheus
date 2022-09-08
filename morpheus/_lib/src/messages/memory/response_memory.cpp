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

#include "morpheus/messages/memory/response_memory.hpp"

#include "morpheus/objects/tensor.hpp"
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/pytypes.h>

#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemory****************************************/
ResponseMemory::ResponseMemory(size_t count) : TensorMemory(count) {}
ResponseMemory::ResponseMemory(size_t count, tensor_map_t &&tensors) : TensorMemory(count, std::move(tensors)) {}

bool ResponseMemory::has_output(const std::string &name) const
{
    return this->has_tensor(name);
}

/****** ResponseMemoryInterfaceProxy *************************/
pybind11::object ResponseMemoryInterfaceProxy::get_output(ResponseMemory &self, const std::string &name)
{
    // Directly return the tensor object
    if (!self.has_tensor(name))
    {
        throw pybind11::key_error();
    }

    return CupyUtil::tensor_to_cupy(self.tensors[name]);
}

TensorObject ResponseMemoryInterfaceProxy::get_output_tensor(ResponseMemory &self, const std::string &name)
{
    // Directly return the tensor object
    if (!self.has_tensor(name))
    {
        throw pybind11::key_error();
    }

    return self.tensors[name];
}
}  // namespace morpheus
