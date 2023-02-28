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

#include "morpheus/messages/memory/tensor_memory.hpp"  // IWYU pragma: associated

#include "morpheus/utilities/cupy_util.hpp"

#include <pybind11/cast.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <map>
#include <sstream>
#include <stdexcept>  // for std::length_error
#include <string>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** TensorMemory****************************************/
TensorMemory::TensorMemory(size_t count) : count(count) {}
TensorMemory::TensorMemory(size_t count, CupyUtil::tensor_map_t&& tensors) : count(count), tensors(std::move(tensors))
{
    check_tensors_length(this->tensors);
}

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

void TensorMemory::check_tensor_length(const TensorObject& tensor)
{
    if (tensor.shape(0) != this->count)
    {
        std::stringstream err_msg;
        err_msg << "The number rows in tensor " << tensor.shape(0) << " does not match TensorMemory.count of "
                << this->count;
        throw std::length_error{err_msg.str()};
    }
}

void TensorMemory::set_tensor(const std::string& name, TensorObject&& tensor)
{
    check_tensor_length(tensor);
    this->tensors.insert_or_assign(name, std::move(tensor));
}

void TensorMemory::check_tensors_length(const CupyUtil::tensor_map_t& tensors)
{
    for (const auto& p : tensors)
    {
        check_tensor_length(p.second);
    }
}

void TensorMemory::set_tensors(CupyUtil::tensor_map_t&& tensors)
{
    check_tensors_length(tensors);
    this->tensors = std::move(tensors);
}

/****** TensorMemoryInterfaceProxy *************************/
std::shared_ptr<TensorMemory> TensorMemoryInterfaceProxy::init(std::size_t count, pybind11::object& tensors)
{
    if (tensors.is_none())
    {
        return std::make_shared<TensorMemory>(count);
    }
    else
    {
        return std::make_shared<TensorMemory>(
            count, std::move(CupyUtil::cupy_to_tensors(tensors.cast<CupyUtil::py_tensor_map_t>())));
    }
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
    self.set_tensors(CupyUtil::cupy_to_tensors(tensors));
}

pybind11::object TensorMemoryInterfaceProxy::get_tensor(TensorMemory& self, const std::string name)
{
    if (!self.has_tensor(name))
    {
        throw pybind11::key_error{};
    }

    return CupyUtil::tensor_to_cupy(self.tensors[name]);
}

void TensorMemoryInterfaceProxy::set_tensor(TensorMemory& self,
                                            const std::string name,
                                            const pybind11::object& cupy_tensor)
{
    self.set_tensor(name, CupyUtil::cupy_to_tensor(cupy_tensor));
}

}  // namespace morpheus
