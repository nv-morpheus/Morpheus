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

from unittest import mock

import pytest
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

from _utils.llm import execute_node
from _utils.llm import mk_mock_langchain_tool
from _utils.llm import mk_mock_openai_response
from morpheus.llm import LLMNodeBase
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode


def test_constructor(mock_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert isinstance(node, LLMNodeBase)


def test_get_input_names(mock_agent_executor: mock.MagicMock):
    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert node.get_input_names() == ["prompt"]


@pytest.mark.parametrize(
    "values,arun_return,expected_output,expected_calls",
    [({
        'prompt': "prompt1"
    }, list(range(3)), list(range(3)), [mock.call(prompt="prompt1")]),
     ({
         'a': ['b', 'c', 'd'], 'c': ['d', 'e', 'f'], 'e': ['f', 'g', 'h']
     },
      list(range(3)), [list(range(3))] * 3,
      [mock.call(a='b', c='d', e='f'), mock.call(a='c', c='e', e='g'), mock.call(a='d', c='f', e='h')])],
    ids=["not-lists", "all-lists"])
def test_execute(
    mock_agent_executor: mock.MagicMock,
    values: dict,
    arun_return: list,
    expected_output: list,
    expected_calls: list[mock.call],
):
    # Tests the execute method of the LangChainAgentNode with a mocked agent_executor
    mock_agent_executor.arun.return_value = arun_return

    node = LangChainAgentNode(agent_executor=mock_agent_executor)
    assert execute_node(node, **values) == expected_output
    mock_agent_executor.arun.assert_has_calls(expected_calls)


def test_execute_tools(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock]):
    # Tests the execute method of the LangChainAgentNode with a a mocked tools and chat completion
    (_, mock_async_client) = mock_chat_completion
    chat_responses = [
        # Responses for first input
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: Yes!',

  # Responses for second input
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: No!'
    ]
    mock_responses = [mk_mock_openai_response([response]) for response in chat_responses]
    mock_async_client.chat.completions.create.side_effect = mock_responses

    llm_chat = ChatOpenAI(model="fake-model", openai_api_key="fake-key")

    mock_tool1 = mk_mock_langchain_tool(["lizard", "frog", "snake"])
    mock_tool2 = mk_mock_langchain_tool(["Tuesday", "Thursday", "Saturday"])

    tools = [
        Tool(name="Tool1",
             func=mock_tool1.run,
             coroutine=mock_tool1.arun,
             description="useful for when you need to know the name of a reptile"),
        Tool(name="Tool2",
             func=mock_tool2.run,
             coroutine=mock_tool2.arun,
             description="useful for when you need to know the day of the week")
    ]

    agent = initialize_agent(tools,
                             llm_chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             early_stopping_method="generate",
                             return_intermediate_steps=False)

    node = LangChainAgentNode(agent_executor=agent)
    assert execute_node(node, input=["input1", "input2"]) == ["Yes!", "No!"]


def test_execute_error(mock_chat_completion: tuple[mock.MagicMock, mock.MagicMock]):
    # Tests the execute method of the LangChainAgentNode with a a mocked tools and chat completion
    (_, mock_async_client) = mock_chat_completion
    chat_responses = [
        'I should check Tool1\nAction: Tool1\nAction Input: "name a reptile"',
        'I should check Tool2\nAction: Tool2\nAction Input: "name of a day of the week"',
        'Observation: Answer: Yes!\nI now know the final answer.\nFinal Answer: Yes!'
    ]
    mock_responses = [mk_mock_openai_response([response]) for response in chat_responses]
    mock_async_client.chat.completions.create.side_effect = mock_responses

    llm_chat = ChatOpenAI(model="fake-model", openai_api_key="fake-key")

    mock_tool1 = mk_mock_langchain_tool(["lizard"])
    mock_tool2 = mk_mock_langchain_tool(RuntimeError("unittest"))

    tools = [
        Tool(name="Tool1",
             func=mock_tool1.run,
             coroutine=mock_tool1.arun,
             description="useful for when you need to know the name of a reptile"),
        Tool(name="Tool2",
             func=mock_tool2.run,
             coroutine=mock_tool2.arun,
             description="useful for when you need to test tool errors")
    ]

    agent = initialize_agent(tools,
                             llm_chat,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True,
                             handle_parsing_errors=True,
                             early_stopping_method="generate",
                             return_intermediate_steps=False)

    node = LangChainAgentNode(agent_executor=agent)
    assert execute_node(node, input="input1") == "Error running agent: unittest"
