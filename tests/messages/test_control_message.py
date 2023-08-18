#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cudf

import morpheus.messages as messages


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_init():
    control_message_one = messages.ControlMessage()  # noqa: F841
    control_message_two = messages.ControlMessage({"test": "test"})  # noqa: F841

@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_tasks():
    message = messages.ControlMessage()
    assert len(message.tasks) == 0

    # Ensure a single task can be read
    message = messages.ControlMessage()
    message.add_task("type_a", { "key_x": "value_x" })

    assert len(message.tasks) == 1
    assert "type_a" in message.tasks
    assert len(message.tasks["type_a"]) == 1
    assert message.tasks["type_a"][0]["key_x"] == "value_x"

    # Ensure multiple task types of different types can be read
    message = messages.ControlMessage()
    message.add_task("type_a", { "key_x": "value_x" })
    message.add_task("type_b", { "key_y": "value_y" })
    assert len(message.tasks) == 2

    assert "type_a" in message.tasks
    assert len(message.tasks["type_a"]) == 1
    assert message.tasks["type_a"][0]["key_x"] == "value_x"

    assert "type_b" in message.tasks
    assert len(message.tasks["type_b"]) == 1
    assert message.tasks["type_b"][0]["key_y"] == "value_y"

    # Ensure multiple task types of the same type can be read
    message = messages.ControlMessage()
    message.add_task("type_a", { "key_x": "value_x" })
    message.add_task("type_a", { "key_y": "value_y" })
    assert len(message.tasks) == 1

    assert "type_a" in message.tasks
    assert len(message.tasks["type_a"]) == 2
    assert message.tasks["type_a"][0]["key_x"] == "value_x"
    assert message.tasks["type_a"][1]["key_y"] == "value_y"

    # Ensure the underlying tasks cannot are not modified
    message = messages.ControlMessage()
    tasks = message.tasks
    tasks["type_a"] = [{"key_x", "value_x"}]
    assert len(message.tasks) == 0

@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_tasks_a():
    # Ensure multiple task types of the same type can be read
    message = messages.ControlMessage()
    message.add_task("type_a", { "key_x": "value_x" })
    message.add_task("type_a", { "key_y": "value_y" })
    assert len(message.tasks) == 2

    assert message.tasks[0].type == "type_a"
    assert "key_x" in message.tasks[0].value
    assert message.tasks[0].value["key_x"] == "value_x"

    assert message.tasks[1].type == "type_a"
    assert "key_y" in message.tasks[0].value
    assert message.tasks[1].value["key_y"] == "value_y"


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_tasks_b():
    message = messages.ControlMessage()
    message.add_task("type_a", { "key_x": "value_x" })
    message.add_task("type_a", { "key_y": "value_y" })
    assert len(message.tasks) == 1

    assert "type_a" in message.tasks
    assert len(message.tasks["type_a"]) == 2
    assert message.tasks["type_a"][0]["key_x"] == "value_x"
    assert message.tasks["type_a"][1]["key_y"] == "value_y"


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_get():
    raw_control_message = messages.ControlMessage({
        "test": "test_rcm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })
    control_message = messages.ControlMessage({
        "test": "test_cm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })

    assert "test" not in raw_control_message.config()
    assert (raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert (control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set():
    raw_control_message = messages.ControlMessage()
    control_message = messages.ControlMessage()

    raw_control_message.config({
        "test": "test_rcm", "tasks": [{
            "type": "load", "properties": {
                "loader_id": "payload"
            }
        }]
    })
    control_message.config({"test": "test_cm", "tasks": [{"type": "load", "properties": {"loader_id": "payload"}}]})

    assert "test" not in raw_control_message.config()
    assert (raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert (control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set_and_get_payload():
    df = cudf.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [True, False, True, False, True]
    })

    msg = messages.ControlMessage()
    payload = messages.MessageMeta(df)
    msg.payload(payload)

    payload2 = msg.payload()
    assert payload2 is not None
    assert payload.df == payload2.df


if (__name__ == "__main__"):
    test_control_message_init()
    test_control_message_get()
    test_control_message_set()
    test_control_message_set_and_get_payload()
