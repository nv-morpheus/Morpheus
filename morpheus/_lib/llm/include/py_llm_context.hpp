/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/llm/llm_context.hpp"

#include <pybind11/pytypes.h>

namespace morpheus::llm {

class PyLLMContext : public LLMContext
{
  public:
    pybind11::object all_outputs() const;
    pybind11::object get_py_input() const;
    pybind11::object get_py_input(const std::string& node_name) const;
    pybind11::object get_py_inputs() const;

    pybind11::object view_outputs() const;

    void set_output(pybind11::object outputs);
    void set_output(const std::string& output_name, pybind11::object output);
};

}  // namespace morpheus::llm
