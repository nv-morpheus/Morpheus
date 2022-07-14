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

#include <morpheus/messages/memory/inference_memory.hpp>
#include <morpheus/messages/memory/inference_memory_fil.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <pybind11/pytypes.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryFIL****************************************/
InferenceMemoryFIL::InferenceMemoryFIL(size_t count, TensorObject input__0, TensorObject seq_ids) :
  InferenceMemory(count)
{
    this->tensors["input__0"] = std::move(input__0);
    this->tensors["seq_ids"]  = std::move(seq_ids);
}

const TensorObject &InferenceMemoryFIL::get_input__0() const
{
    auto found = this->tensors.find("input__0");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'input__0' not found in memory");
    }

    return found->second;
}

void InferenceMemoryFIL::set_input__0(TensorObject input__0)
{
    this->tensors["input__0"] = std::move(input__0);
}

const TensorObject &InferenceMemoryFIL::get_seq_ids() const
{
    auto found = this->tensors.find("seq_ids");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'seq_ids' not found in memory");
    }

    return found->second;
}

void InferenceMemoryFIL::set_seq_ids(TensorObject seq_ids)
{
    this->tensors["seq_ids"] = std::move(seq_ids);
}
/****** InferenceMemoryFILInterfaceProxy *************************/
std::shared_ptr<InferenceMemoryFIL> InferenceMemoryFILInterfaceProxy::init(cudf::size_type count,
                                                                           pybind11::object input__0,
                                                                           pybind11::object seq_ids)
{
    // Convert the cupy arrays to tensors
    return std::make_shared<InferenceMemoryFIL>(
        count, std::move(CupyUtil::cupy_to_tensor(input__0)), std::move(CupyUtil::cupy_to_tensor(seq_ids)));
}

std::size_t InferenceMemoryFILInterfaceProxy::count(InferenceMemoryFIL &self)
{
    return self.count;
}

TensorObject InferenceMemoryFILInterfaceProxy::get_tensor(InferenceMemoryFIL &self, const std::string &name)
{
    return self.tensors[name];
}

pybind11::object InferenceMemoryFILInterfaceProxy::get_input__0(InferenceMemoryFIL &self)
{
    return CupyUtil::tensor_to_cupy(self.get_input__0());
}

void InferenceMemoryFILInterfaceProxy::set_input__0(InferenceMemoryFIL &self, pybind11::object cupy_values)
{
    self.set_input__0(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryFILInterfaceProxy::get_seq_ids(InferenceMemoryFIL &self)
{
    return CupyUtil::tensor_to_cupy(self.get_seq_ids());
}

void InferenceMemoryFILInterfaceProxy::set_seq_ids(InferenceMemoryFIL &self, pybind11::object cupy_values)
{
    return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
