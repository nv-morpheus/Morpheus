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

#include <nlohmann/json.hpp>

namespace nlohmann {
// NOLINTBEGIN(readability-identifier-naming)

/*
    Derived class from basic_json to allow for custom type names. Use this if the return type would always be an object
   (i.e. dict[str, Any] in python)
*/
// NLOHMANN_BASIC_JSON_TPL_DECLARATION
class json_dict : public basic_json<>
{};

/*
    Derived class from basic_json to allow for custom type names. Use this if the return type would always be a list
   (i.e. list[Any] in python)
*/
class json_list : public basic_json<>
{};

// NOLINTEND(readability-identifier-naming)
}  // namespace nlohmann
