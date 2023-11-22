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

#include "morpheus/llm/llm_context.hpp"
#include "morpheus/llm/llm_engine.hpp"
#include "morpheus/llm/llm_lambda_node.hpp"
#include "morpheus/llm/llm_task.hpp"
#include "morpheus/types.hpp"

#include <gtest/gtest.h>
#include <mrc/channel/forward.hpp>
#include <mrc/coroutines/event.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <mrc/coroutines/task.hpp>

#include <atomic>
#include <coroutine>
#include <cstdint>
#include <memory>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

TEST_CLASS(LLMEngine);

TEST_F(TestLLMEngine, AsyncTest)
{
    coroutines::Event e{};

    std::atomic<uint64_t> counter{0};

    llm::LLMEngine engine;

    auto start_fn = [&]() -> coroutines::Task<int> {
        // co_await tp.schedule();
        co_await e;

        counter++;

        co_return 123;
    };

    auto test_fn = [&](int i) -> coroutines::Task<int> {
        // co_await tp.schedule();
        co_await e;

        counter++;

        co_return i + 1;
    };

    engine.add_node("start", {}, llm::make_lambda_node(start_fn));
    engine.add_node("test", {{"/start"}}, llm::make_lambda_node(test_fn));

    auto context = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);

    auto return_val = engine.execute(context);

    // Start the coroutine
    return_val.resume();

    EXPECT_FALSE(return_val.is_ready());
    EXPECT_EQ(counter, 0);

    // Release the event
    e.set();

    auto out_context = mrc::coroutines::sync_wait(return_val);

    EXPECT_TRUE(return_val.is_ready());
    EXPECT_EQ(counter, 2);

    EXPECT_EQ(out_context->view_outputs()["start"], 123);
    EXPECT_EQ(out_context->view_outputs()["test"], 124);
}
