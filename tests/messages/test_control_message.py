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
import morpheus._lib.messages as _messages
import morpheus.messages as messages


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_init():
    raw_control_message_one = _messages.MessageControl()
    raw_control_message_two = _messages.MessageControl({"test": "test"})

    control_message_one = messages.MessageControl()
    control_message_two = messages.MessageControl({"test": "test"})


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_get():
    raw_control_message = _messages.MessageControl(
        {
            "test": "test_rcm",
            "tasks": [
                {
                    "type": "load",
                    "properties": {
                        "loader_id": "payload"
                    }
                }
            ]
        }
    )
    control_message = messages.MessageControl(
        {
            "test": "test_cm",
            "tasks": [
                {
                    "type": "load",
                    "properties": {
                        "loader_id": "payload"
                    }
                }
            ]
        }
    )

    assert "test" not in raw_control_message.config()
    assert(raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert(control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set():
    raw_control_message = _messages.MessageControl()
    control_message = messages.MessageControl()

    raw_control_message.config({
        "test": "test_rcm",
        "tasks": [
            {
                "type": "load",
                "properties": {
                    "loader_id": "payload"
                }
            }
        ]
    })
    control_message.config({
        "test": "test_cm",
        "tasks": [
            {
                "type": "load",
                "properties": {
                    "loader_id": "payload"
                }
            }
        ]
    })

    assert "test" not in raw_control_message.config()
    assert (raw_control_message.has_task("load"))

    assert "test" not in control_message.config()
    assert(control_message.has_task("load"))


@pytest.mark.usefixtures("config_only_cpp")
def test_control_message_set_and_get_payload():
    df = cudf.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col3': ['a', 'b', 'c', 'd', 'e'],
        'col4': [True, False, True, False, True]
    })

    msg = messages.MessageControl()
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
