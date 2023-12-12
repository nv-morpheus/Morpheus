# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.haystack_agent_node import HaystackAgentNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.concat_df import concat_dataframes

logger = logging.getLogger(__name__)


def _build_langchain_agent_executor(model_name: str):
    from langchain import OpenAI
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    from langchain.agents import load_tools

    llm = OpenAI(model=model_name, temperature=0)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_haystack_agent(model_name: str):

    from haystack.agents import Agent
    from haystack.agents import Tool
    from haystack.agents.base import ToolsManager
    from haystack.nodes import PromptNode
    from haystack.nodes import PromptTemplate
    from haystack.nodes.retriever.web import WebRetriever
    from haystack.pipelines import WebQAPipeline

    search_key = os.environ.get("SERPERDEV_API_KEY")
    if not search_key:
        raise ValueError("Ensure to configure the SERPERDEV_API_KEY environment variable.")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Ensure to configure the OPENAI_API_KEY environment variable.")

    web_prompt_node = PromptNode(
        model_name,
        api_key=openai_key,
        max_length=256,
        default_prompt_template="deepset/question-answering",
    )

    calc_prompt_node = PromptNode(model_name,
                                  api_key=openai_key,
                                  default_prompt_template="""
        Calculate the result of the following mathematical expression:

        Expression: ({query})
        """)

    web_retriever = WebRetriever(api_key=search_key)
    web_qa_pipeline = WebQAPipeline(retriever=web_retriever, prompt_node=web_prompt_node)

    prompt_template = PromptTemplate("deepset/zero-shot-react")
    prompt_node = PromptNode(model_name,
                             api_key=os.environ.get("OPENAI_API_KEY"),
                             max_length=512,
                             stop_words=["Observation:"])

    web_qa_tool = Tool(
        name="Search",
        pipeline_or_node=web_qa_pipeline,
        description="Useful when you need to search for answers online.",
        output_variable="results",
    )

    calc_tool = Tool(name="Calculator",
                     pipeline_or_node=calc_prompt_node,
                     description="Useful when you need to math calculations.")

    agent = Agent(prompt_node=prompt_node,
                  prompt_template=prompt_template,
                  tools_manager=ToolsManager([web_qa_tool, calc_tool]))

    return agent


def _build_engine(model_name: str, llm_orch: str) -> LLMEngine:

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    agent_node = None

    if llm_orch == "langchain":
        agent_node = LangChainAgentNode(agent_executor=_build_langchain_agent_executor(model_name=model_name))
    elif llm_orch == "haystack":
        agent_node = HaystackAgentNode(agent=_build_haystack_agent(model_name=model_name))
    else:
        raise RuntimeError(f"LLM orchestration framework '{llm_orch}' is not supported yet.")

    engine.add_node("agent", inputs=[("/extracter")], node=agent_node)

    engine.add_task_handler(inputs=["/agent"], handler=SimpleTaskHandler())

    return engine


def pipeline(num_threads: int,
             pipeline_batch_size: int,
             model_max_batch_size: int,
             model_name: str,
             repeat_count: int,
             llm_orch: str) -> float:
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    source_dfs = [
        cudf.DataFrame({
            "questions": [
                "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
                "Who is the 7th president of United States?"
            ]
        })
    ]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=repeat_count))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name, llm_orch=llm_orch)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    pipe.run()

    messages = sink.get_messages()
    responses = concat_dataframes(messages)

    logger.info("Pipeline complete. Received %s responses:\n%s", len(messages), responses['response'])

    return start_time
