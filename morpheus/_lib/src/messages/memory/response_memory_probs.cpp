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

#include <morpheus/messages/memory/response_memory.hpp>
#include <morpheus/messages/memory/response_memory_probs.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>

#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <glog/logging.h>
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace morpheus {
/****** Component public implementations *******************/
/****** ResponseMemoryProbs****************************************/
ResponseMemoryProbs::ResponseMemoryProbs(size_t count, TensorObject probs) : ResponseMemory(count)
{
    this->outputs["probs"] = std::move(probs);
}

ResponseMemoryProbs::ResponseMemoryProbs(size_t count, std::map<std::string, TensorObject> &&outputs) :
  ResponseMemory(count)
{
    CHECK(outputs.find("probs") != outputs.end()) << "ResponseMemoryProbs requeires an output named \"probs\"";
    this->outputs.merge(std::move(outputs));
}

ResponseMemoryProbs::ResponseMemoryProbs(ResponseMemory &&other) : ResponseMemory{other.count}
{
    CHECK(other.has_output("probs")) << "ResponseMemoryProbs requeires an output named \"probs\"";
    outputs     = std::move(other.outputs);
    other.count = 0;
}

const TensorObject &ResponseMemoryProbs::get_probs() const
{
    auto found = this->outputs.find("probs");
    if (found == this->outputs.end())
    {
        throw std::runtime_error("Tensor: 'probs' not found in memory");
    }

    return found->second;
}

void ResponseMemoryProbs::set_probs(TensorObject probs)
{
    this->outputs["probs"] = std::move(probs);
}

/****** ResponseMemoryProbsInterfaceProxy *************************/
std::shared_ptr<ResponseMemoryProbs> ResponseMemoryProbsInterfaceProxy::init(cudf::size_type count,
                                                                             pybind11::object probs)
{
    // Conver the cupy arrays to tensors
    return std::make_shared<ResponseMemoryProbs>(count, std::move(CupyUtil::cupy_to_tensor(probs)));
}

std::size_t ResponseMemoryProbsInterfaceProxy::count(ResponseMemoryProbs &self)
{
    return self.count;
}

pybind11::object ResponseMemoryProbsInterfaceProxy::get_probs(ResponseMemoryProbs &self)
{
    return CupyUtil::tensor_to_cupy(self.get_probs());
}

void ResponseMemoryProbsInterfaceProxy::set_probs(ResponseMemoryProbs &self, pybind11::object cupy_values)
{
    self.set_probs(CupyUtil::cupy_to_tensor(cupy_values));
}
}  // namespace morpheus
