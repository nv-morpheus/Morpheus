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

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace morpheus::llm {

class MORPHEUS_EXPORT LLMNode : public LLMNodeBase
{
  public:
    LLMNode();
    ~LLMNode() override;

    virtual std::shared_ptr<LLMNodeRunner> add_node(std::string name,
                                                    input_mapping_t inputs,
                                                    std::shared_ptr<LLMNodeBase> node,
                                                    bool is_output = false);

    std::vector<std::string> get_input_names() const override;

    const std::vector<std::string>& get_output_node_names() const;

    size_t node_count() const;

    Task<std::shared_ptr<LLMContext>> execute(std::shared_ptr<LLMContext> context) override;

  private:
    std::vector<std::shared_ptr<LLMNodeRunner>> m_child_runners;

    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_node_names;  // Names of nodes to be used as the output
};

}  // namespace morpheus::llm
