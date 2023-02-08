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


import morpheus.messages as messages
import morpheus._lib.messages as _messages


def test_control_message_init():
    raw_control_message_one = _messages.MessageControl()
    raw_control_message_two = _messages.MessageControl({"test": "test"})

    control_message_one = messages.MessageControl()
    control_message_two = messages.MessageControl({"test": "test"})


def test_control_message_get():
    raw_control_message = _messages.MessageControl({"test": "test_rcm"})
    control_message = messages.MessageControl({"test": "test_cm"})

    assert "test" in raw_control_message.message()
    assert raw_control_message.message()["test"] == "test_rcm"

    assert "test" in control_message.message()
    assert control_message.message()["test"] == "test_cm"


def test_control_message_set():
    raw_control_message = _messages.MessageControl()
    control_message = messages.MessageControl()

    raw_control_message.message({"test": "test_rcm"})
    control_message.message({"test": "test_cm"})

    assert "test" in raw_control_message.message()
    assert raw_control_message.message()["test"] == "test_rcm"

    assert "test" in control_message.message()
    assert control_message.message()["test"] == "test_cm"


if (__name__ == "__main__"):
    test_control_message_init()
    test_control_message_get()
    test_control_message_set()
