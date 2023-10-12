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
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_task_handler.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/types.hpp"

#include <memory>
#include <vector>

namespace morpheus::llm {

class MORPHEUS_EXPORT LLMEngine : public LLMNode
{
  public:
    LLMEngine();
    ~LLMEngine() override;

    virtual void add_task_handler(user_input_mappings_t inputs, std::shared_ptr<LLMTaskHandler> task_handler);

    virtual Task<std::vector<std::shared_ptr<ControlMessage>>> run(std::shared_ptr<ControlMessage> input_message);

  private:
    Task<std::vector<std::shared_ptr<ControlMessage>>> handle_tasks(std::shared_ptr<LLMContext> context);

    std::vector<std::shared_ptr<LLMTaskHandlerRunner>> m_task_handlers;
};

}  // namespace morpheus::llm
