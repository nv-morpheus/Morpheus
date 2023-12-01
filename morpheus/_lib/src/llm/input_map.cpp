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

#include "morpheus/llm/input_map.hpp"

#include "morpheus/llm/llm_node_runner.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace morpheus::llm {

InputMap::InputMap() = default;

InputMap::InputMap(std::string external_name, std::string internal_name) :
  external_name(std::move(external_name)),
  internal_name(std::move(internal_name))
{
    CHECK_NE(this->internal_name, "") << "Cannot have an empty internal name. Use UserInputMapping for placeholders.";
    CHECK_NE(this->internal_name, "-") << "Cannot use placeholders. Use UserInputMapping for placeholders.";
}

UserInputMapping::UserInputMapping() = default;

UserInputMapping::UserInputMapping(std::string external_name, std::string internal_name) :
  external_name(std::move(external_name)),
  internal_name(std::move(internal_name))
{}

UserInputMapping::UserInputMapping(const std::shared_ptr<LLMNodeRunner>& runner) :
  external_name(runner->name()),
  internal_name("-")
{}

UserInputMapping::UserInputMapping(std::tuple<std::string, std::string>&& tuple) :
  external_name(std::move(std::get<0>(tuple))),
  internal_name(std::move(std::get<1>(tuple)))
{}

}  // namespace morpheus::llm
