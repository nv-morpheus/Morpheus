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

import pandas as pd
import pytest
from langchain import OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools

import cudf

from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes


def _build_agent_executor(model_name: str):

    llm = OpenAI(model=model_name, temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

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


def _run_pipeline(config: Config, questions: list[str], model_name: str = "test_model") -> pd.DataFrame:
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
    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.run()

    result_df = concat_dataframes(sink.get_messages())

    return result_df


@pytest.mark.usefixtures("openai")
@pytest.mark.usefixtures("openai_api_key")
@pytest.mark.usefixtures("serpapi_api_key")
@pytest.mark.use_python
def test_agents_simple_pipe_integration_openai(config: Config):
    questions = ["Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"]
    result_df = _run_pipeline(config, questions=questions, model_name="gpt-3.5-turbo-instruct")

    assert len(result_df.columns) == 2
    assert any(result_df.columns == ["questions", "response"])
    assert float(result_df.response.iloc[0]) >= 3.7
