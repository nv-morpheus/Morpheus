/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_morpheus.hpp"

#include <filesystem>
#include <stdexcept>

namespace morpheus::test {

std::filesystem::path get_morpheus_root()
{
    auto root = std::getenv("MORPHEUS_ROOT");

    if (root == nullptr)
    {
        throw std::runtime_error("MORPHEUS_ROOT env variable is not set");
    }

    return std::filesystem::path{root};
}

}  // namespace morpheus::test
