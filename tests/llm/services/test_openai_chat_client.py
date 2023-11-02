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

import asyncio
from unittest import mock

import pytest

from morpheus.llm.services.llm_service import LLMClient
from morpheus.llm.services.openai_chat_service import OpenAIChatClient


def test_constructor(mock_chat_completion: mock.MagicMock):
    client = OpenAIChatClient(model_name="test_model")
    assert isinstance(client, LLMClient)
    mock_chat_completion.assert_not_called()


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "input_dict, set_assistant, expected_messages",
    [({
        "prompt": "test_prompt", "assistant": "assistant_response"
    },
      True,
      [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "test_prompt"
      }, {
          "role": "assistant", "content": "assistant_response"
      }]),
     ({
         "prompt": "test_prompt"
     },
      False, [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "test_prompt"
      }])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate(mock_chat_completion: mock.MagicMock,
                  use_async: bool,
                  input_dict: dict[str, str],
                  set_assistant: bool,
                  expected_messages: list[dict],
                  temperature: int):
    client = OpenAIChatClient(model_name="test_model", set_assistant=set_assistant, temperature=temperature)
    if use_async:
        results = asyncio.run(client.generate_async(input_dict))
        mock_chat_completion.acreate.assert_called_once_with(model="test_model",
                                                             messages=expected_messages,
                                                             temperature=temperature)

    else:
        results = client.generate(input_dict)
        mock_chat_completion.create.assert_called_once_with(model="test_model",
                                                            messages=expected_messages,
                                                            temperature=temperature)

    assert results == "test_output"


@pytest.mark.parametrize("use_async", [True, False])
@pytest.mark.parametrize(
    "inputs, set_assistant, expected_messages",
    [({
        "prompt": ["prompt1", "prompt2"], "assistant": ["assistant1", "assistant2"]
    },
      True,
      [[{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt1"
      }, {
          "role": "assistant", "content": "assistant1"
      }],
       [{
           "role": "system", "content": "You are a helpful assistant."
       }, {
           "role": "user", "content": "prompt2"
       }, {
           "role": "assistant", "content": "assistant2"
       }]]),
     ({
         "prompt": ["prompt1", "prompt2"]
     },
      False,
      [[{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt1"
      }], [{
          "role": "system", "content": "You are a helpful assistant."
      }, {
          "role": "user", "content": "prompt2"
      }]])])
@pytest.mark.parametrize("temperature", [0, 1, 2])
def test_generate_batch(mock_chat_completion: mock.MagicMock,
                        use_async: bool,
                        inputs: dict[str, list[str]],
                        set_assistant: bool,
                        expected_messages: list[list[dict]],
                        temperature: int):
    client = OpenAIChatClient(model_name="test_model", set_assistant=set_assistant, temperature=temperature)

    expected_results = ["test_output" for _ in range(len(inputs["prompt"]))]
    expected_calls = [
        mock.call(model="test_model", messages=messages, temperature=temperature) for messages in expected_messages
    ]

    if use_async:
        results = asyncio.run(client.generate_batch_async(inputs))
        mock_chat_completion.acreate.assert_has_calls(expected_calls, any_order=False)
    else:
        results = client.generate_batch(inputs)
        mock_chat_completion.create.assert_has_calls(expected_calls, any_order=False)

    assert results == expected_results


@pytest.mark.parametrize("completion", [{
    "choices": []
}, {
    "choices": [{}]
}, {
    "choices": [{
        "message": {}
    }]
}],
                         ids=["no_choices", "no_message", "no_content"])
def test_generate_invalid_completions(mock_chat_completion: mock.MagicMock, completion: dict):
    mock_chat_completion.create.return_value = completion

    client = OpenAIChatClient(model_name="test_model")

    with pytest.raises(ValueError):
        client.generate({"prompt": "test_prompt"})
