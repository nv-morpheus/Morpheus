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

#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/llm/input_map.hpp"
#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_lambda_node.hpp"
#include "morpheus/llm/llm_node.hpp"
#include "morpheus/llm/llm_node_runner.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/llm/utils.hpp"
#include "morpheus/types.hpp"

#include <gtest/gtest.h>
#include <mrc/channel/forward.hpp>
#include <mrc/coroutines/sync_wait.hpp>

#include <coroutine>
#include <memory>
#include <stdexcept>
#include <string>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

TEST_CLASS(LLMUtils);

TEST_F(TestLLMUtils, IsValidNameEmpty)
{
    EXPECT_FALSE(llm::is_valid_node_name(""));
}

TEST_F(TestLLMUtils, IsValidNameSpecialCharacters)
{
    EXPECT_FALSE(llm::is_valid_node_name("my_name*"));
    EXPECT_FALSE(llm::is_valid_node_name("my_name/stuff"));
    EXPECT_FALSE(llm::is_valid_node_name("my_name[0]"));
}

TEST_F(TestLLMUtils, IsValidNameSpace)
{
    EXPECT_FALSE(llm::is_valid_node_name("my_name other_name"));
}

TEST_F(TestLLMUtils, ProcessInputNamesSingleString)
{
    auto input_names = std::vector<std::string>{"input1"};
    auto user_inputs = llm::user_input_mappings_t{{"/ext1"}};

    auto returned = llm::process_input_names(user_inputs, input_names);

    EXPECT_EQ(returned.size(), 1);
    EXPECT_EQ(returned[0].external_name, "/ext1");
    EXPECT_EQ(returned[0].internal_name, "input1");
}

TEST_F(TestLLMUtils, ProcessInputNamesCountMismatch)
{
    auto input_names = std::vector<std::string>{"input1", "input2"};
    auto user_inputs = llm::user_input_mappings_t{llm::InputMap{"/ext1", "input1"}, llm::InputMap{"/ext2", "input1"}};

    EXPECT_THROW(llm::process_input_names(user_inputs, input_names), std::invalid_argument);
}
