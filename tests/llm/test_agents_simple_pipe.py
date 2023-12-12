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

import os
import re
import sys
import types
from unittest import mock

import langchain
import pytest
from langchain.agents import AgentType
from langchain.agents import initialize_agent

import cudf

from _utils import assert_results
from _utils import import_module
from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes


@pytest.mark.usefixtures("restore_sys_path")
@pytest.fixture(name="load_tools_mod")
def load_tools_mod_fixture():
    """
    We need to import the `langchain.agents.load_tools` module in order to mock the `SerpAPIWrapper` class that is
    imported there. However this is more difficult than it sounds, because `langchain.agents.load_tools` is a module
    containing a function of the same name. In `agents/__init__.py` we have:
    ```
    from langchain.agents.load_tools import load_tools
    ```

    Making it very difficult to import the module not the function.
    """
    mod_path = os.path.join(os.path.dirname(langchain.__file__), 'agents/load_tools.py')
    (mod_name, mod) = import_module(mod_path=mod_path)
    yield mod

    sys.modules.pop(mod_name, None)


@pytest.fixture(name="questions")
def questions_fixture():
    return ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]


def _build_agent_executor(model_name: str):

    llm = langchain.OpenAI(model=model_name, temperature=0)

    tools = langchain.agents.load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_engine(model_name: str):

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=_build_agent_executor(model_name=model_name)))

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def _run_pipeline(config: Config,
                  questions: list[str],
                  model_name: str = "test_model",
                  expected_df: cudf.DataFrame = None) -> InMemorySinkStage:
    """
    Loosely patterned after `examples/llm/completion`
    """
    source_df = cudf.DataFrame({"questions": questions})

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"]}}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=[source_df]))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    if expected_df is not None:
        sink = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))
    else:
        sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    return sink


@pytest.mark.usefixtures("openai", "openai_api_key", "serpapi_api_key")
@pytest.mark.use_python
def test_agents_simple_pipe_integration_openai(config: Config, questions: list[str]):
    sink = _run_pipeline(config, questions=questions, model_name="gpt-3.5-turbo-instruct")

    result_df = concat_dataframes(sink.get_messages())
    assert len(result_df.columns) == 2
    assert sorted(result_df.columns) == ["questions", "response"]

    response_txt = result_df.response.iloc[0]
    response_match = re.match(r".*(\d+\.\d+)\.?$", response_txt)
    assert response_match is not None
    assert float(response_match.group(1)) >= 3.7


@pytest.mark.usefixtures("openai", "restore_environ")
@pytest.mark.use_python
@mock.patch("langchain.OpenAI._agenerate", autospec=True)  # autospec is needed as langchain will inspect the function
def test_agents_simple_pipe(mock_openai_agenerate: mock.AsyncMock,
                            load_tools_mod: types.ModuleType,
                            config: Config,
                            questions: list[str]):
    os.environ.update({'OPENAI_API_KEY': 'test_api_key', 'SERPAPI_API_KEY': 'test_api_key'})

    # mocking the SerpAPIWrapper imported in load_tools
    mock_serpapi_aresults = mock.AsyncMock()
    load_tools_mod.SerpAPIWrapper.aresults = mock_serpapi_aresults

    from langchain.schema import Generation
    from langchain.schema import LLMResult

    model_name = "test_model"

    async def mock_agenerate_impl(*args, **kwargs):
        nonlocal model_name
        generations = [[
            Generation(text="approximately 3.99", generation_info={
                'finish_reason': 'stop', 'logprobs': None
            })
        ]]
        return LLMResult(generations=generations, llm_output={'token_usage': {}, 'model_name': model_name})

    mock_openai_agenerate.side_effect = mock_agenerate_impl

    mock_serpapi_aresults.return_value = {
        'answer_box': {
            'answer': '25 years', 'link': 'http://unit.test', 'people_also_search_for': []
        },
        'inline_people_also_search_for': [],
        'knowledge_graph': {},
        'organic_results': [],
        'pagination': {},
        'related_questions': [],
        'related_searches': [],
        'search_information': {},
        'search_metadata': {},
        'search_parameters': {},
        'serpapi_pagination': None
    }

    expected_df = cudf.DataFrame({'questions': questions, 'response': ['approximately 3.99']})

    sink = _run_pipeline(config, questions=questions, model_name=model_name, expected_df=expected_df)

    assert len(mock_openai_agenerate.mock_calls) == len(questions)
    assert len(mock_serpapi_aresults.mock_calls) == len(questions)

    assert_results(sink.get_results())
