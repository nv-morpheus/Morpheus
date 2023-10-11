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

#include "morpheus/llm/llm_node_runner.hpp"

#include "morpheus/llm/llm_context.hpp"

#include <glog/logging.h>
#include <mrc/coroutines/task.hpp>

namespace morpheus::llm {
LLMNodeRunner::LLMNodeRunner(std::string name, input_map_t inputs, std::shared_ptr<LLMNodeBase> node) :
  m_name(std::move(name)),
  m_inputs(std::move(inputs)),
  m_node(std::move(node))
{
    // TODO(MDD): Check that the input map is valid

    // Get the inputs of the current node
    auto input_names = m_node->get_input_names();

    // Replace any placeholders with the real node input name
    for (size_t i = 0; i < m_inputs.size(); ++i)
    {
        const auto& node_name  = m_inputs[i].node_name;
        const auto& input_name = m_inputs[i].input_name;

        // Check that the input name and node names are valid
        CHECK_EQ(node_name.find("*"), std::string::npos) << "Invalid node name '" << node_name << "'";
        CHECK_EQ(input_name.find("*"), std::string::npos) << "Invalid input_name '" << input_name << "'";

        CHECK_EQ(node_name.find("-"), std::string::npos) << "Invalid node name '" << node_name << "'";
        CHECK_EQ(input_name.find("-"), std::string::npos) << "Invalid input_name '" << input_name << "'";

        // Determine if the inputs are coming from a parent node or a sibling node
        if (m_inputs[i].input_name[0] == '/')
        {
            m_sibling_input_names.push_back(m_inputs[i].input_name);
        }
        else
        {
            m_parent_input_names.push_back(m_inputs[i].input_name);
        }
    }
}

LLMNodeRunner::~LLMNodeRunner() = default;

Task<std::shared_ptr<LLMContext>> LLMNodeRunner::execute(std::shared_ptr<LLMContext> context)
{
    // Create a new context
    auto child_context = context->push(m_name, m_inputs);

    // Also need error handling here
    auto returned_context = co_await m_node->execute(child_context);

    // Call pop to apply the outputs to the parent context
    child_context->pop();

    co_return returned_context;
}

const std::string& LLMNodeRunner::name() const
{
    return m_name;
}

const input_map_t& LLMNodeRunner::inputs() const
{
    return m_inputs;
}

const std::vector<std::string>& LLMNodeRunner::sibling_input_names() const
{
    return m_sibling_input_names;
}

const std::vector<std::string>& LLMNodeRunner::parent_input_names() const
{
    return m_parent_input_names;
}

}  // namespace morpheus::llm
