# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

from _utils.llm import execute_node
from morpheus.llm import LLMNodeBase
from morpheus.llm.nodes.llm_generate_node import LLMGenerateNode
from morpheus.llm.services.llm_service import LLMClient


def test_constructor():
    node = LLMGenerateNode(llm_client=mock.MagicMock(LLMClient))
    assert isinstance(node, LLMNodeBase)


def test_get_input_names():
    node = LLMGenerateNode(llm_client=mock.MagicMock(LLMClient))
    assert node.get_input_names() == ["prompt"]


def test_execute():
    expected_output = ["response1", "response2"]
    mock_client = mock.MagicMock(LLMClient)
    mock_client.return_value = mock_client
    mock_client.generate_batch_async = mock.AsyncMock(return_value=expected_output.copy())

    node = LLMGenerateNode(llm_client=mock_client)
    assert execute_node(node, prompt=["prompt1", "prompt2"]) == expected_output
    mock_client.generate_batch_async.assert_called_once_with(["prompt1", "prompt2"])
