/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nlohmann/json.hpp"
#include "test_messages.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <memory>

using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestControlMessage, InitializationTest)
{
    auto msg_one = ControlMessage();

    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    auto msg_two = ControlMessage(config);

    ASSERT_EQ(msg_two.has_task("load"), true);
}

TEST_F(TestControlMessage, SetMessageTest)
{
    auto msg = ControlMessage();

    ASSERT_EQ(msg.config().contains("nope"), false);

    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    msg.config(config);

    ASSERT_EQ(msg.has_task("load"), true);
}

TEST_F(TestControlMessage, TaskTest)
{
    auto msg_infer = ControlMessage();
    auto msg_train = ControlMessage();

    ASSERT_EQ(msg_infer.config().contains("some_value"), false);

    auto config = nlohmann::json();
    nlohmann::json task_properties;
    task_properties = {
        {"loader_id", "payload"},
        {"strategy", "aggregate"},
    };

    config["type"]  = "inference";
    config["tasks"] = {{{"type", "load"}, {"properties", task_properties}}};

    msg_infer.config(config);

    ASSERT_EQ(msg_infer.has_task("load"), true);
    ASSERT_EQ(msg_infer.has_task("inference"), false);
    ASSERT_EQ(msg_infer.has_task("training"), false);
    ASSERT_EQ(msg_infer.has_task("custom"), false);

    msg_infer.add_task("inference", {});
    ASSERT_EQ(msg_infer.has_task("inference"), true);

    msg_infer.remove_task("inference");
    ASSERT_EQ(msg_infer.has_task("inference"), false);

    ASSERT_THROW(msg_infer.add_task("training", {}), std::runtime_error);

    config["type"] = "training";
    msg_train.config(config);
    msg_train.add_task("training", {});
    ASSERT_EQ(msg_train.has_task("training"), true);
    msg_train.remove_task("training");
    ASSERT_EQ(msg_train.has_task("training"), false);

    ASSERT_THROW(msg_train.add_task("inference", {}), std::runtime_error);

    msg_train.add_task("custom", {});
    ASSERT_EQ(msg_train.has_task("custom"), true);
    msg_train.remove_task("custom");
    ASSERT_EQ(msg_train.has_task("custom"), false);
}

TEST_F(TestControlMessage, PayloadTest)
{
    auto msg = ControlMessage();

    ASSERT_EQ(msg.payload(), nullptr);

    auto null_payload = std::shared_ptr<MessageMeta>(nullptr);

    msg.payload(null_payload);

    ASSERT_EQ(msg.payload(), null_payload);

    auto data_payload = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    msg.payload(data_payload);

    ASSERT_EQ(msg.payload(), data_payload);
}