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

#include "morpheus/messages/memory/inference_memory_nlp.hpp"

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/types.hpp"  // for tensor_map_t
#include "morpheus/utilities/cupy_util.hpp"

#include <cudf/types.hpp>  // for size_type
#include <pybind11/pytypes.h>

#include <cstddef>
#include <map>        // this->tensors is a map
#include <stdexcept>  // for runtime_error
#include <utility>    // for move, pair

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryNLP ****************************************/
InferenceMemoryNLP::InferenceMemoryNLP(std::size_t count,
                                       TensorObject input_ids,
                                       TensorObject input_mask,
                                       TensorObject seq_ids) :
  InferenceMemory(count)
{
    this->tensors["input_ids"]  = std::move(input_ids);
    this->tensors["input_mask"] = std::move(input_mask);
    this->tensors["seq_ids"]    = std::move(seq_ids);
}

const TensorObject& InferenceMemoryNLP::get_input_ids() const
{
    auto found = this->tensors.find("input_ids");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'input_ids' not found in memory");
    }

    return found->second;
}

void InferenceMemoryNLP::set_input_ids(TensorObject input_ids)
{
    this->tensors["input_ids"] = std::move(input_ids);
}

const TensorObject& InferenceMemoryNLP::get_input_mask() const
{
    auto found = this->tensors.find("input_mask");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'input_mask' not found in memory");
    }

    return found->second;
}

void InferenceMemoryNLP::set_input_mask(TensorObject input_mask)
{
    this->tensors["input_mask"] = std::move(input_mask);
}

const TensorObject& InferenceMemoryNLP::get_seq_ids() const
{
    auto found = this->tensors.find("seq_ids");
    if (found == this->tensors.end())
    {
        throw std::runtime_error("Tensor: 'seq_ids' not found in memory");
    }

    return found->second;
}

void InferenceMemoryNLP::set_seq_ids(TensorObject seq_ids)
{
    this->tensors["seq_ids"] = std::move(seq_ids);
}

/****** InferenceMemoryNLPInterfaceProxy *************************/
std::shared_ptr<InferenceMemoryNLP> InferenceMemoryNLPInterfaceProxy::init(cudf::size_type count,
                                                                           pybind11::object input_ids,
                                                                           pybind11::object input_mask,
                                                                           pybind11::object seq_ids)
{
    // Convert the cupy arrays to tensors
    return std::make_shared<InferenceMemoryNLP>(count,
                                                std::move(CupyUtil::cupy_to_tensor(input_ids)),
                                                std::move(CupyUtil::cupy_to_tensor(input_mask)),
                                                std::move(CupyUtil::cupy_to_tensor(seq_ids)));
}

std::size_t InferenceMemoryNLPInterfaceProxy::count(InferenceMemoryNLP& self)
{
    return self.count;
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_ids(InferenceMemoryNLP& self)
{
    return CupyUtil::tensor_to_cupy(self.get_input_ids());
}

void InferenceMemoryNLPInterfaceProxy::set_input_ids(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    self.set_input_ids(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_mask(InferenceMemoryNLP& self)
{
    return CupyUtil::tensor_to_cupy(self.get_input_mask());
}

void InferenceMemoryNLPInterfaceProxy::set_input_mask(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    return self.set_input_mask(CupyUtil::cupy_to_tensor(cupy_values));
}

pybind11::object InferenceMemoryNLPInterfaceProxy::get_seq_ids(InferenceMemoryNLP& self)
{
    return CupyUtil::tensor_to_cupy(self.get_seq_ids());
}

void InferenceMemoryNLPInterfaceProxy::set_seq_ids(InferenceMemoryNLP& self, pybind11::object cupy_values)
{
    return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
