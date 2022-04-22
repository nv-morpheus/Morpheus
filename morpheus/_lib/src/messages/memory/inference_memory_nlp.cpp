/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <morpheus/messages/memory/inference_memory_nlp.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/utilities/cupy_util.hpp>

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <cstddef>

namespace morpheus {
    /****** Component public implementations *******************/
    /****** InferenceMemoryNLP ****************************************/
    InferenceMemoryNLP::InferenceMemoryNLP(std::size_t count,
                                           neo::TensorObject input_ids,
                                           neo::TensorObject input_mask,
                                           neo::TensorObject seq_ids) :
            InferenceMemory(count) {
        this->inputs["input_ids"] = std::move(input_ids);
        this->inputs["input_mask"] = std::move(input_mask);
        this->inputs["seq_ids"] = std::move(seq_ids);
    }

    const neo::TensorObject &InferenceMemoryNLP::get_input_ids() const {
        auto found = this->inputs.find("input_ids");
        if (found == this->inputs.end()) {
            throw std::runtime_error("Tensor: 'input_ids' not found in memory");
        }

        return found->second;
    }

    void InferenceMemoryNLP::set_input_ids(neo::TensorObject input_ids) {
        this->inputs["input_ids"] = std::move(input_ids);
    }

    const neo::TensorObject &InferenceMemoryNLP::get_input_mask() const {
        auto found = this->inputs.find("input_mask");
        if (found == this->inputs.end()) {
            throw std::runtime_error("Tensor: 'input_mask' not found in memory");
        }

        return found->second;
    }

    void InferenceMemoryNLP::set_input_mask(neo::TensorObject input_mask) {
        this->inputs["input_mask"] = std::move(input_mask);
    }

    const neo::TensorObject &InferenceMemoryNLP::get_seq_ids() const {
        auto found = this->inputs.find("seq_ids");
        if (found == this->inputs.end()) {
            throw std::runtime_error("Tensor: 'seq_ids' not found in memory");
        }

        return found->second;
    }

    void InferenceMemoryNLP::set_seq_ids(neo::TensorObject seq_ids) {
        this->inputs["seq_ids"] = std::move(seq_ids);
    }

    /****** InferenceMemoryNLPInterfaceProxy *************************/
    std::shared_ptr<InferenceMemoryNLP>
    InferenceMemoryNLPInterfaceProxy::init(cudf::size_type count, pybind11::object input_ids,
                                           pybind11::object input_mask, pybind11::object seq_ids)  {
        // Convert the cupy arrays to tensors
        return std::make_shared<InferenceMemoryNLP>(count,
                                                    std::move(CupyUtil::cupy_to_tensor(input_ids)),
                                                    std::move(CupyUtil::cupy_to_tensor(input_mask)),
                                                    std::move(CupyUtil::cupy_to_tensor(seq_ids)));
    }

    std::size_t InferenceMemoryNLPInterfaceProxy::count(InferenceMemoryNLP& self) {
        return self.count;
    }

    pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_ids(InferenceMemoryNLP &self) {
        return CupyUtil::tensor_to_cupy(self.get_input_ids());
    }

    void InferenceMemoryNLPInterfaceProxy::set_input_ids(InferenceMemoryNLP &self, pybind11::object cupy_values) {
        self.set_input_ids(CupyUtil::cupy_to_tensor(cupy_values));
    }

    pybind11::object InferenceMemoryNLPInterfaceProxy::get_input_mask(InferenceMemoryNLP &self) {
        return CupyUtil::tensor_to_cupy(self.get_input_mask());
    }

    void InferenceMemoryNLPInterfaceProxy::set_input_mask(InferenceMemoryNLP &self, pybind11::object cupy_values) {
        return self.set_input_mask(CupyUtil::cupy_to_tensor(cupy_values));
    }

    pybind11::object InferenceMemoryNLPInterfaceProxy::get_seq_ids(InferenceMemoryNLP &self) {
        return CupyUtil::tensor_to_cupy(self.get_seq_ids());
    }

    void InferenceMemoryNLPInterfaceProxy::set_seq_ids(InferenceMemoryNLP &self, pybind11::object cupy_values) {
        return self.set_seq_ids(CupyUtil::cupy_to_tensor(cupy_values));
    }
}