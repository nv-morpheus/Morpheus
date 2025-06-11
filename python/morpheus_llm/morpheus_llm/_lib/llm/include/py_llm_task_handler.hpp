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
#include "morpheus_llm/llm/llm_task_handler.hpp"

#include "morpheus/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace morpheus::llm {

class PyLLMTaskHandler : public LLMTaskHandler
{
  public:
    using LLMTaskHandler::LLMTaskHandler;

    ~PyLLMTaskHandler() override;

    std::vector<std::string> get_input_names() const override;

    Task<LLMTaskHandler::return_t> try_handle(std::shared_ptr<LLMContext> context) override;
};

}  // namespace morpheus::llm
