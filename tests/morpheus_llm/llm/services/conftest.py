# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Override fixtures from parent setting autouse to True


@pytest.fixture(name="openai", autouse=True, scope='session')
def openai_fixture(openai):
    """
    All of the tests in this subdir require openai
    """
    yield openai


@pytest.fixture(name="langchain_nvidia_ai_endpoints", autouse=True, scope='session')
def langchain_nvidia_ai_endpoints_fixture(langchain_nvidia_ai_endpoints):
    """
    All of the tests in this subdir require langchain_nvidia_ai_endpoints
    """
    yield langchain_nvidia_ai_endpoints


@pytest.fixture(name="mock_chat_completion", autouse=True)
def mock_chat_completion_fixture(mock_chat_completion):
    yield mock_chat_completion
