# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import time

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents.agent import AgentExecutor
from langchain.llms.openai import OpenAIChat

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm.nodes.extracter_node import ExtracterNode
from morpheus.llm.nodes.langchain_agent_node import LangChainAgentNode
from morpheus.llm.task_handlers.simple_task_handler import SimpleTaskHandler
from morpheus.messages import ControlMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage

logger = logging.getLogger(__name__)


def _build_agent_executor(model_name: str) -> AgentExecutor:

    llm = OpenAIChat(model_name=model_name, model_kwargs={"temperature": 0.0}, client=None)

    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    agent_executor = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    return agent_executor


def _build_engine(model_name: str) -> LLMEngine:

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    engine.add_node("agent",
                    inputs=[("/extracter")],
                    node=LangChainAgentNode(agent_executor=_build_agent_executor(model_name=model_name)))

    engine.add_task_handler(inputs=["/extracter"], handler=SimpleTaskHandler())

    return engine


def pipeline(num_threads: int, pipeline_batch_size: int, model_max_batch_size: int, model_name: str) -> float:
    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["question"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(KafkaSourceStage(config, bootstrap_servers="auto", input_topic=["input"]))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    # pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine(model_name=model_name)))

    sink = pipe.add_stage(InMemorySinkStage(config))

    # pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    logger.info("Pipeline running. Waiting for input from Kafka...")

    pipe.run()

    logger.info("Pipeline complete. Received %s responses", len(sink.get_messages()))

    return start_time
