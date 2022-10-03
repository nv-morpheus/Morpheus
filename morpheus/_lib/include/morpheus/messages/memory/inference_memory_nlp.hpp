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

#include "morpheus/messages/memory/inference_memory.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cudf/types.hpp>  // for size_type
#include <pybind11/pytypes.h>

#include <cstddef>
#include <memory>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryNLP**********************************/
/**
 * TODO(Documentation)
 */
class InferenceMemoryNLP : public InferenceMemory
{
  public:
    InferenceMemoryNLP(std::size_t count, TensorObject input_ids, TensorObject input_mask, TensorObject seq_ids);

    /**
     * TODO(Documentation)
     */
    const TensorObject& get_input_ids() const;

    /**
     * TODO(Documentation)
     */
    const TensorObject& get_input_mask() const;

    /**
     * TODO(Documentation)
     */
    const TensorObject& get_seq_ids() const;

    /**
     * TODO(Documentation)
     */
    void set_input_ids(TensorObject input_ids);

    /**
     * TODO(Documentation)
     */
    void set_input_mask(TensorObject input_mask);

    /**
     * TODO(Documentation)
     */
    void set_seq_ids(TensorObject seq_ids);
};

/****** InferenceMemoryNLPInterfaceProxy********************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct InferenceMemoryNLPInterfaceProxy
{
    /**
     * @brief Create and initialize an InferenceMemoryNLP object, and return a shared pointer to the result.
     */
    static std::shared_ptr<InferenceMemoryNLP> init(cudf::size_type count,
                                                    pybind11::object input_ids,
                                                    pybind11::object input_mask,
                                                    pybind11::object seq_ids);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(InferenceMemoryNLP& self);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_input_ids(InferenceMemoryNLP& self);

    /**
     * TODO(Documentation)
     */
    static void set_input_ids(InferenceMemoryNLP& self, pybind11::object cupy_values);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_input_mask(InferenceMemoryNLP& self);

    /**
     * TODO(Documentation)
     */
    static void set_input_mask(InferenceMemoryNLP& self, pybind11::object cupy_values);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_seq_ids(InferenceMemoryNLP& self);

    /**
     * TODO(Documentation)
     */
    static void set_seq_ids(InferenceMemoryNLP& self, pybind11::object cupy_values);
};
#pragma GCC visibility pop
}  // namespace morpheus
