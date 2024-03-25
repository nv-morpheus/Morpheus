/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "py_llm_context.hpp"

#include "morpheus/utilities/string_util.hpp"  // for MORPHEUS_CONCAT_STR

#include <pybind11/pybind11.h>
#include <pymrc/utilities/json_values.hpp>
#include <pymrc/utils.hpp>  // for cast_from_json

#include <memory>

namespace morpheus::llm {
namespace py = pybind11;

py::object PyLLMContext::all_outputs() const
{
    return m_outputs.to_python();
}

py::object PyLLMContext::get_py_input() const
{
    if (m_inputs.size() > 1)
    {
        throw std::runtime_error(
            "PyLLMContext::get_input() called on a context with multiple inputs. Use get_input(input_name) instead.");
    }

    return this->get_py_input(m_inputs[0].internal_name);
}

py::object PyLLMContext::get_py_input(const std::string& node_name) const
{
    if (node_name[0] == '/')
    {
        auto py_dict = m_outputs.to_python().cast<py::dict>();

        try
        {
            return py_dict[node_name.c_str()];
        } catch (const py::key_error& e)
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR("Input '" << node_name << "' not found in the output map"));
        }
    }
    // TODO: Make this a method on the parent class
    auto found = std::find_if(m_inputs.begin(), m_inputs.end(), [&node_name](const auto& map_iterator) {
        return map_iterator.internal_name == node_name;
    });

    if (found == m_inputs.end())
    {
        std::stringstream error_msg;
        error_msg << "Input '" << node_name << "' not found in the input list.";

        if (!m_inputs.empty())
        {
            error_msg << " Available inputs are:";
            for (const auto& input : m_inputs)
            {
                error_msg << " '" << input.internal_name << "'";
            }
        }
        else
        {
            error_msg << " Input list is empty.";
        }

        throw std::runtime_error(error_msg.str());
    }

    auto& input_name = found->external_name;

    // Get the value from a parent output
    auto py_parent = std::dynamic_pointer_cast<PyLLMContext>(m_parent);
    if (py_parent)
    {
        return py_parent->get_py_input(input_name);
    }

    auto json_input = m_parent->get_input(input_name);
    return mrc::pymrc::cast_from_json(json_input).cast<py::dict>();
}

py::object PyLLMContext::get_py_inputs() const
{
    py::dict inputs;
    for (const auto& in_map : m_inputs)
    {
        inputs[in_map.internal_name.c_str()] = this->get_py_input(in_map.internal_name);
    }

    return inputs;
}

py::object PyLLMContext::view_outputs() const
{
    return this->all_outputs();
}

void PyLLMContext::set_output(py::object outputs)
{
    mrc::pymrc::JSONValues json_values(outputs);
    LLMContext::set_output(std::move(json_values));
}

void PyLLMContext::set_output(const std::string& output_name, py::object output)
{
    mrc::pymrc::JSONValues json_value(output);
    LLMContext::set_output(output_name, std::move(json_value));
}

}  // namespace morpheus::llm
