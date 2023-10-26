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
#include "morpheus/types.hpp"

#include <gtest/gtest.h>
#include <mrc/channel/forward.hpp>
#include <mrc/coroutines/sync_wait.hpp>
#include <nlohmann/json.hpp>

#include <coroutine>
#include <memory>
#include <stdexcept>
#include <string>

using namespace morpheus;
using namespace morpheus::test;
using namespace mrc;

TEST_CLASS(LLMContext);

TEST_F(TestLLMContext, InitializationTest)
{
    llm::LLMContext ctx_1;

    nlohmann::json task_dict;
    task_dict = {
        {"task_type", "dictionary"},
        {"model_name", "test"},
    };

    llm::LLMContext ctx_2{llm::LLMTask{"template", task_dict}, nullptr};
    ASSERT_EQ(ctx_2.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_2.task().get("model_name"), "test");

    // llm::LLMContext context_2{llm::LLMTask{}, nullptr};

    nlohmann::json msg_config;
    msg_config["tasks"] = {{{"type", "llm_engine"}, {"properties", {{"type", "template"}, {"properties", task_dict}}}}};

    auto msg = std::make_shared<ControlMessage>(msg_config);

    llm::LLMContext ctx_3{llm::LLMTask{}, msg};
    ASSERT_EQ(ctx_3.message()->has_task("llm_engine"), true);

    llm::LLMContext ctx_4{llm::LLMTask{"template", task_dict}, msg};
    ASSERT_EQ(ctx_4.message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx_4.task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_4.task().get("model_name"), "test");

    auto parent_ctx = std::make_shared<llm::LLMContext>(llm::LLMTask{"template", task_dict}, msg);
    auto inputs     = llm::input_mappings_t{{"/ext1", "input1"}};
    llm::LLMContext ctx_5{parent_ctx, "child", inputs};
    ASSERT_EQ(ctx_5.input_map()[0].external_name, "/ext1");
    ASSERT_EQ(ctx_5.input_map()[0].internal_name, "input1");
    ASSERT_EQ(ctx_5.parent()->message()->has_task("llm_engine"), true);
    ASSERT_EQ(ctx_5.parent()->task().get("task_type"), "dictionary");
    ASSERT_EQ(ctx_5.parent()->task().get("model_name"), "test");
    ASSERT_EQ(ctx_5.name(), "child");
    ASSERT_EQ(ctx_5.full_name(), "/child");
}

TEST_F(TestLLMContext, OutputTest)
{
    llm::LLMContext ctx_1{llm::LLMTask{}, nullptr};
    nlohmann::json outputs;
    outputs["input1"] = {{"key1", "val1"}, {"key2", "val2"}};
    ctx_1.set_output(outputs);

    ASSERT_EQ(ctx_1.view_outputs()["input1"]["key1"], "val1");
    ASSERT_EQ(ctx_1.view_outputs()["input1"]["key2"], "val2");

    llm::LLMContext ctx_2{llm::LLMTask{}, nullptr};
    ctx_2.set_output("input1", outputs["input1"]);
    ASSERT_EQ(ctx_2.view_outputs()["input1"]["key1"], "val1");
    ASSERT_EQ(ctx_2.view_outputs()["input1"]["key2"], "val2");
}

TEST_F(TestLLMContext, PushPopTest)
{
    auto parent_context = std::make_shared<llm::LLMContext>(llm::LLMTask{}, nullptr);
    auto inputs         = llm::input_mappings_t{{"/ext1", "input1"}};
    llm::LLMContext child_context{parent_context, "child", inputs};

    ASSERT_EQ(child_context.name(), "child");
    ASSERT_EQ(child_context.input_map()[0].external_name, "/ext1");
    ASSERT_EQ(child_context.input_map()[0].internal_name, "input1");

    nlohmann::json outputs;
    outputs["input1"] = {{"key1", "val1"}, {"key2", "val2"}};
    child_context.set_output(outputs);
    ASSERT_EQ(child_context.view_outputs()["input1"]["key1"], "val1");
    ASSERT_EQ(child_context.view_outputs()["input1"]["key2"], "val2");

    child_context.pop();
    ASSERT_EQ(parent_context->view_outputs()["child"]["input1"]["key1"], "val1");
    ASSERT_EQ(parent_context->view_outputs()["child"]["input1"]["key2"], "val2");

    std::cerr << parent_context->view_outputs();
}
