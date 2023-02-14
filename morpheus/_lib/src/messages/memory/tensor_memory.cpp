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

#include "morpheus/messages/memory/tensor_memory.hpp"

#include "morpheus/utilities/cupy_util.hpp"

#include <pybind11/pybind11.h>  // for key_error & object

#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/
TensorMemory::TensorMemory(size_t count) : count(count) {}
TensorMemory::TensorMemory(size_t count, CupyUtil::tensor_map_t&& tensors) : count(count), tensors(std::move(tensors))
{}

bool TensorMemory::has_tensor(const std::string& name) const
{
    return this->tensors.find(name) != this->tensors.end();
}

CupyUtil::tensor_map_t TensorMemory::copy_tensor_ranges(const std::vector<std::pair<TensorIndex, TensorIndex>>& ranges,
                                                        size_t num_selected_rows) const
{
    CupyUtil::tensor_map_t tensors;
    for (const auto& p : this->tensors)
    {
        tensors.insert(std::pair{p.first, p.second.copy_rows(ranges, num_selected_rows)});
    }

    return tensors;
}

/****** TensorMemoryInterfaceProxy *************************/
std::shared_ptr<TensorMemory> TensorMemoryInterfaceProxy::init(std::size_t count, CupyUtil::py_tensor_map_t tensors)
{
    return std::make_shared<TensorMemory>(count, std::move(CupyUtil::cupy_to_tensors(tensors)));
}

std::size_t TensorMemoryInterfaceProxy::get_count(TensorMemory& self)
{
    return self.count;
}

CupyUtil::py_tensor_map_t TensorMemoryInterfaceProxy::get_tensors(TensorMemory& self)
{
    return CupyUtil::tensors_to_cupy(self.tensors);
}

void TensorMemoryInterfaceProxy::set_tensors(TensorMemory& self, CupyUtil::py_tensor_map_t tensors)
{
    self.tensors = std::move(CupyUtil::cupy_to_tensors(tensors));
}

const TensorObject& TensorMemoryInterfaceProxy::get_tensor_object(TensorMemory& self, const std::string& name)
{
    const auto tensor_itr = self.tensors.find(name);
    if (tensor_itr == self.tensors.end())
    {
        throw pybind11::key_error{};
    }

    return tensor_itr->second;
}

pybind11::object TensorMemoryInterfaceProxy::get_tensor(TensorMemory& self, const std::string name)
{
    return CupyUtil::tensor_to_cupy(TensorMemoryInterfaceProxy::get_tensor_object(self, name));
}

void TensorMemoryInterfaceProxy::set_tensor(TensorMemory& self,
                                            const std::string name,
                                            const pybind11::object& cupy_tensor)
{
    self.tensors.insert_or_assign(name, CupyUtil::cupy_to_tensor(cupy_tensor));
}

}  // namespace morpheus
