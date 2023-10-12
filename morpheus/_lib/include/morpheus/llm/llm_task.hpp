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

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

namespace morpheus::llm {

struct MORPHEUS_EXPORT LLMTask
{
    LLMTask();
    LLMTask(std::string task_type, nlohmann::json task_dict);

    ~LLMTask();

    std::string task_type;

    size_t size() const;

    const nlohmann::json& get(const std::string& key) const;

    void set(const std::string& key, nlohmann::json&& value);

    nlohmann::json task_dict;
};

}  // namespace morpheus::llm
