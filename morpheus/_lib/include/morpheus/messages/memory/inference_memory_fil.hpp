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

#include <morpheus/messages/memory/inference_memory.hpp>
#include <morpheus/objects/python_data_table.hpp>
#include <morpheus/objects/tensor.hpp>
#include <morpheus/objects/tensor_object.hpp>

#include <cudf/io/types.hpp>

#include <pybind11/pybind11.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace morpheus {
/****** Component public implementations *******************/
/****** InferenceMemoryFIL****************************************/
class InferenceMemoryFIL : public InferenceMemory
{
  public:
    InferenceMemoryFIL(size_t count, TensorObject input__0, TensorObject seq_ids);

    /**
     * TODO(Documentation)
     */
    const TensorObject& get_input__0() const;

    /**
     * TODO(Documentation)
     */
    const TensorObject& get_seq_ids() const;

    /**
     * TODO(Documentation)
     */
    void set_input__0(TensorObject input_ids);

    /**
     * TODO(Documentation)
     */
    void set_seq_ids(TensorObject input_mask);
};

/****** InferenceMemoryFILInterfaceProxy *************************/
#pragma GCC visibility push(default)
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct InferenceMemoryFILInterfaceProxy
{
    /**
     * @brief Create and initialize an InferenceMemoryFIL object, and return a shared pointer to the result.
     */
    static std::shared_ptr<InferenceMemoryFIL> init(cudf::size_type count,
                                                    pybind11::object input__0,
                                                    pybind11::object seq_ids);

    /**
     * TODO(Documentation)
     */
    static std::size_t count(InferenceMemoryFIL& self);

    /**
     * TODO(Documentation)
     */
    static TensorObject get_tensor(InferenceMemoryFIL& self, const std::string& name);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_input__0(InferenceMemoryFIL& self);

    /**
     * TODO(Documentation)
     */
    static void set_input__0(InferenceMemoryFIL& self, pybind11::object cupy_values);

    /**
     * TODO(Documentation)
     */
    static pybind11::object get_seq_ids(InferenceMemoryFIL& self);

    /**
     * TODO(Documentation)
     */
    static void set_seq_ids(InferenceMemoryFIL& self, pybind11::object cupy_values);
};
#pragma GCC visibility pop
}  // namespace morpheus
