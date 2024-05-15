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
import openai
import os
from unittest import mock

import pytest

from _utils.llm import mk_mock_openai_response
from morpheus.llm.services.openai_chat_service import OpenAIChatService


@pytest.mark.parametrize("api_key", ["12345", None])
@pytest.mark.parametrize("base_url", ["http://test.openai.com/v1", None])
@pytest.mark.parametrize("max_retries", [5, 10])
def test_constructor(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                     api_key: str,
                     base_url: str,
                     max_retries: int):
    OpenAIChatService(api_key=api_key, base_url=base_url).get_client(model_name="test_model", max_retries=max_retries)

    for mock_client in mock_chat_completion:
        mock_client.assert_called_once_with(api_key=api_key, base_url=base_url, max_retries=max_retries)


@pytest.mark.parametrize("max_retries", [5, 10])
def test_constructor_default_service_constructor(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                                                 max_retries: int):
    OpenAIChatService().get_client(model_name="test_model", max_retries=max_retries)

    for mock_client in mock_chat_completion:
        mock_client.assert_called_once_with(api_key=None, base_url=None, max_retries=max_retries)


@pytest.mark.usefixtures("openai")
@pytest.mark.parametrize('use_env', [True, False])
def test_constructor_api_key_base_url(use_env: bool):

    if use_env:
        env_api_key = "env-12345"
        env_base_url = "http://env.openai.com/v1/"
        os.environ["OPENAI_API_KEY"] = env_api_key
        os.environ["OPENAI_BASE_URL"] = env_base_url

        # Test when api_key and base_url are not passed
        client = OpenAIChatService().get_client(model_name="test_model")
        assert client._client.api_key == env_api_key
        assert str(client._client.base_url) == env_base_url

        # Test when api_key and base_url are passed
        arg_api_key = "arg-12345"
        arg_base_url = "http://arg.openai.com/v1/"
        client = OpenAIChatService(api_key=arg_api_key, base_url=arg_base_url).get_client(model_name="test_model")
        assert client._client.api_key == arg_api_key
        assert str(client._client.base_url) == arg_base_url
    else:
        os.environ.pop("OPENAI_API_KEY")
        os.environ.pop("OPENAI_BASE_URL")
        # Test when api_key and base_url are not passed
        with pytest.raises(openai.OpenAIError) as excinfo:
            client = OpenAIChatService().get_client(model_name="test_model")

        assert "api_key client option must be set" in str(excinfo.value)

        # Test when only api_key is passed
        arg_api_key = "arg-12345"
        client = OpenAIChatService(api_key=arg_api_key).get_client(model_name="test_model")
        assert client._client.api_key == arg_api_key
        assert str(client._client.base_url) == "https://api.openai.com/v1/"

        # Test when both api_key and base_url are passed
        arg_base_url = "http://arg.openai.com/v1/"
        client = OpenAIChatService(api_key=arg_api_key, base_url=arg_base_url).get_client(model_name="test_model")
        assert client._client.api_key == arg_api_key
        assert str(client._client.base_url) == arg_base_url


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "input_dict, set_assistant, expected_messages",
    [({
        "prompt": "test_prompt", "assistant": "assistant_response"
    },
      True, [{
          "role": "user", "content": "test_prompt"
      }, {
          "role": "assistant", "content": "assistant_response"
      }]), ({
          "prompt": "test_prompt"
      }, False, [{
          "role": "user", "content": "test_prompt"
      }])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                  use_async: bool,
                  input_dict: dict[str, str],
                  set_assistant: bool,
                  expected_messages: list[dict],
                  temperature: int):
    (mock_client, mock_async_client) = mock_chat_completion
    client = OpenAIChatService().get_client(model_name="test_model",
                                            set_assistant=set_assistant,
                                            temperature=temperature)

    if use_async:
        results = asyncio.run(client.generate_async(**input_dict))
        mock_async_client.chat.completions.create.assert_called_once_with(model="test_model",
                                                                          messages=expected_messages,
                                                                          temperature=temperature)
        mock_client.chat.completions.create.assert_not_called()

    else:
        results = client.generate(**input_dict)
        mock_client.chat.completions.create.assert_called_once_with(model="test_model",
                                                                    messages=expected_messages,
                                                                    temperature=temperature)
        mock_async_client.chat.completions.create.assert_not_called()

    assert results == "test_output"


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize("inputs, set_assistant, expected_messages",
                         [({
                             "prompt": ["prompt1", "prompt2"], "assistant": ["assistant1", "assistant2"]
                         },
                           True,
                           [[{
                               "role": "user", "content": "prompt1"
                           }, {
                               "role": "assistant", "content": "assistant1"
                           }], [{
                               "role": "user", "content": "prompt2"
                           }, {
                               "role": "assistant", "content": "assistant2"
                           }]]),
                          ({
                              "prompt": ["prompt1", "prompt2"]
                          },
                           False, [[{
                               "role": "user", "content": "prompt1"
                           }], [{
                               "role": "user", "content": "prompt2"
                           }]])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate_batch(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock],
                        use_async: bool,
                        inputs: dict[str, list[str]],
                        set_assistant: bool,
                        expected_messages: list[list[dict]],
                        temperature: int):
    (mock_client, mock_async_client) = mock_chat_completion
    client = OpenAIChatService().get_client(model_name="test_model",
                                            set_assistant=set_assistant,
                                            temperature=temperature)

    expected_results = ["test_output" for _ in range(len(inputs["prompt"]))]
    expected_calls = [
        mock.call(model="test_model", messages=messages, temperature=temperature) for messages in expected_messages
    ]

    if use_async:
        results = asyncio.run(client.generate_batch_async(inputs))
        mock_async_client.chat.completions.create.assert_has_calls(expected_calls, any_order=False)
        mock_client.chat.completions.create.assert_not_called()

    else:
        results = client.generate_batch(inputs)
        mock_client.chat.completions.create.assert_has_calls(expected_calls, any_order=False)
        mock_async_client.chat.completions.create.assert_not_called()

    assert results == expected_results


@pytest.mark.parametrize("completion", [[], [None]], ids=["no_choices", "no_content"])
@pytest.mark.usefixtures("mock_chat_completion")
def test_extract_completion_errors(completion: list):
    client = OpenAIChatService().get_client(model_name="test_model")
    mock_completion = mk_mock_openai_response(completion)

    with pytest.raises(ValueError):
        client._extract_completion(mock_completion)
