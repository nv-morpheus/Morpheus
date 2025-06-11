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

#pragma once

#include "morpheus_llm/llm/fwd.hpp"
#include "morpheus_llm/llm/input_map.hpp"
#include "py_llm_node_base.hpp"

#include <pybind11/pytypes.h>

#include <map>
#include <memory>
#include <string>

// IWYU pragma: no_include "morpheus_llm/llm/llm_engine.hpp"
// IWYU pragma: no_include "morpheus_llm/llm/llm_node.hpp"

namespace morpheus::llm {

template <class BaseT = LLMNode>
class PyLLMNode : public PyLLMNodeBase<BaseT>
{
  public:
    using PyLLMNodeBase<BaseT>::PyLLMNodeBase;

    std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                            user_input_mappings_t inputs,
                                            std::shared_ptr<LLMNodeBase> node,
                                            bool is_output = false) override;

  private:
    std::map<std::shared_ptr<LLMNodeBase>, pybind11::object> m_py_nodes;
};

}  // namespace morpheus::llm
