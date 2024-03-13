# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
from unittest import mock

import pytest

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.nemo_llm_service import NeMoLLMClient


def test_constructor(mock_nemollm: mock.MagicMock, mock_nemo_service: mock.MagicMock):
    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    assert isinstance(client, LLMClient)
    mock_nemollm.assert_not_called()


def test_get_input_names(mock_nemollm: mock.MagicMock, mock_nemo_service: mock.MagicMock):
    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    assert client.get_input_names() == ["prompt"]
    mock_nemollm.assert_not_called()


def test_generate(mock_nemollm: mock.MagicMock, mock_nemo_service: mock.MagicMock):
    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    assert client.generate({'prompt': "test_prompt"}) == "test_output"
    mock_nemollm.generate_multiple.assert_called_once_with(model="test_model",
                                                           prompts=["test_prompt"],
                                                           return_type="text",
                                                           additional_arg="test_arg")


def test_generate_batch(mock_nemollm: mock.MagicMock, mock_nemo_service: mock.MagicMock):
    mock_nemollm.generate_multiple.return_value = ["output1", "output2"]

    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    assert client.generate_batch({'prompt': ["prompt1", "prompt2"]}) == ["output1", "output2"]
    mock_nemollm.generate_multiple.assert_called_once_with(model="test_model",
                                                           prompts=["prompt1", "prompt2"],
                                                           return_type="text",
                                                           additional_arg="test_arg")


@mock.patch("asyncio.wrap_future")
@mock.patch("asyncio.gather", new_callable=mock.AsyncMock)
def test_generate_async(
        mock_asyncio_gather: mock.AsyncMock,
        mock_asyncio_wrap_future: mock.MagicMock,  # pylint: disable=unused-argument
        mock_nemollm: mock.MagicMock,
        mock_nemo_service: mock.MagicMock):
    mock_asyncio_gather.return_value = [mock.MagicMock()]

    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    results = asyncio.run(client.generate_async({'prompt': "test_prompt"}))
    assert results == "test_output"
    mock_nemollm.generate.assert_called_once_with("test_model",
                                                  "test_prompt",
                                                  return_type="async",
                                                  additional_arg="test_arg")


@mock.patch("asyncio.wrap_future")
@mock.patch("asyncio.gather", new_callable=mock.AsyncMock)
def test_generate_batch_async(
        mock_asyncio_gather: mock.AsyncMock,
        mock_asyncio_wrap_future: mock.MagicMock,  # pylint: disable=unused-argument
        mock_nemollm: mock.MagicMock,
        mock_nemo_service: mock.MagicMock):
    mock_asyncio_gather.return_value = [mock.MagicMock(), mock.MagicMock()]
    mock_nemollm.post_process_generate_response.side_effect = [{"text": "output1"}, {"text": "output2"}]

    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")
    results = asyncio.run(client.generate_batch_async({'prompt': ["prompt1", "prompt2"]}))
    assert results == ["output1", "output2"]
    mock_nemollm.generate.assert_has_calls([
        mock.call("test_model", "prompt1", return_type="async", additional_arg="test_arg"),
        mock.call("test_model", "prompt2", return_type="async", additional_arg="test_arg")
    ])


@mock.patch("asyncio.wrap_future")
@mock.patch("asyncio.gather", new_callable=mock.AsyncMock)
def test_generate_batch_async_error(
        mock_asyncio_gather: mock.AsyncMock,
        mock_asyncio_wrap_future: mock.MagicMock,  # pylint: disable=unused-argument
        mock_nemollm: mock.MagicMock,
        mock_nemo_service: mock.MagicMock):
    mock_asyncio_gather.return_value = [mock.MagicMock(), mock.MagicMock()]
    mock_nemollm.post_process_generate_response.return_value = {"status": "fail", "msg": "unittest"}

    client = NeMoLLMClient(mock_nemo_service, model_name="test_model", additional_arg="test_arg")

    with pytest.raises(RuntimeError, match="unittest"):
        asyncio.run(client.generate_batch_async({'prompt': ["prompt1", "prompt2"]}))
