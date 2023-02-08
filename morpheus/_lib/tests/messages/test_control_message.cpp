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

using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestControlMessage, InitializationTest)
{
    auto msg_one = MessageControl();

    ASSERT_EQ(msg_one.message_type(), MessageControl::ControlMessageType::noop);

    auto config          = nlohmann::json();
    config["some_value"] = "42";

    auto msg_two = MessageControl(config);

    ASSERT_EQ(msg_two.message_type(), MessageControl::ControlMessageType::noop);
    ASSERT_EQ(msg_two.message().contains("some_value"), true);
    ASSERT_EQ(msg_two.message()["some_value"], "42");
}

TEST_F(TestControlMessage, SetMessageTest)
{
    auto msg = MessageControl();

    ASSERT_EQ(msg.message().contains("some_value"), false);

    auto config          = nlohmann::json();
    config["some_value"] = "42";

    msg.message(config);

    ASSERT_EQ(msg.message().contains("some_value"), true);
    ASSERT_EQ(msg.message()["some_value"], "42");
}