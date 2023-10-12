/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "py_llm_lambda_node.hpp"

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node_base.hpp"
#include "morpheus/utilities/json_types.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <mrc/coroutines/task.hpp>  // IWYU pragma: keep
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymrc/utils.hpp>

#include <coroutine>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace morpheus::llm {
namespace py = pybind11;

PyLLMLambdaNode::PyLLMLambdaNode(pybind11::function fn) : LLMNodeBase(), m_fn(std::move(fn))
{
    if (!m_fn)
    {
        throw std::runtime_error("Invalid function. Function cannot be null to a LLMLambdaNode");
    }

    auto inspect               = pybind11::module_::import("inspect");
    auto POSITIONAL_OR_KEYWORD = inspect.attr("Parameter").attr("POSITIONAL_OR_KEYWORD");
    auto KEYWORD_ONLY          = inspect.attr("Parameter").attr("KEYWORD_ONLY");

    // Check the function signature to determine what the inputs and outputs are
    auto sig = inspect.attr("signature")(m_fn);

    for (const auto& item : sig.attr("parameters").attr("items")())
    {
        auto item_tuple = item.cast<py::tuple>();
        auto name       = item_tuple[0].cast<std::string>();
        auto param      = item_tuple[1];

        if (param.attr("kind").equal(POSITIONAL_OR_KEYWORD) || param.attr("kind").equal(KEYWORD_ONLY))
        {
            m_input_names.push_back(std::move(name));
        }
        else
        {
            throw std::runtime_error(MORPHEUS_CONCAT_STR(
                "Invalid argument '" << name << "' in wrapped function: " << py::str(m_fn)
                                     << ". Function arguments must either KEYWORD_ONLY or POSITIONAL_OR_KEYWORD"));
        }
    }
}

PyLLMLambdaNode::~PyLLMLambdaNode() = default;

std::vector<std::string> PyLLMLambdaNode::get_input_names() const
{
    return m_input_names;
}

Task<std::shared_ptr<LLMContext>> PyLLMLambdaNode::execute(std::shared_ptr<LLMContext> context)
{
    // Get the inputs. This will be a dictionary
    auto inputs = context->get_inputs();

    // Grab the GIL
    pybind11::gil_scoped_acquire gil;

    // Convert to python dictionary
    auto py_inputs = mrc::pymrc::cast_from_json(std::move(inputs));

    // Call the function
    auto py_return_val = m_fn(**py_inputs);

    // Convert back to JSON
    auto return_val = mrc::pymrc::cast_from_pyobject(py_return_val);

    // Set the object back into the context outputs
    context->set_output(std::move(return_val));

    co_return context;
}
}  // namespace morpheus::llm
