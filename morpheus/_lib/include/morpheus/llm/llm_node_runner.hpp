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

#pragma once

#include "morpheus/export.h"
#include "morpheus/llm/fwd.hpp"
#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_node_base.hpp"

#include <coroutine>
#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

class MORPHEUS_EXPORT LLMNodeRunner
{
  public:
    LLMNodeRunner(std::string name, input_mapping_t inputs, std::shared_ptr<LLMNodeBase> node);

    ~LLMNodeRunner();

    virtual Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context);

    const std::string& name() const;

    const input_mapping_t& inputs() const;

    const std::vector<std::string>& sibling_input_names() const;

    const std::vector<std::string>& parent_input_names() const;

  private:
    std::string m_name;
    input_mapping_t m_inputs;
    std::shared_ptr<LLMNodeBase> m_node;

    std::vector<std::string> m_sibling_input_names;
    std::vector<std::string> m_parent_input_names;
};

}  // namespace morpheus::llm
