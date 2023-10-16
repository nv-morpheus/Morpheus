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
import asyncio
import logging
import os
import pickle
import time
import typing

import click
import mrc
import mrc.core.operators as ops
import pandas as pd
import pymilvus
import requests_cache
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.llm import LLMEngine
from morpheus.llm import LLMLambdaNode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.output.write_to_vector_db import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from ..common.extracter_node import ExtracterNode
from ..common.llm_engine_stage import LLMEngineStage
from ..common.simple_task_handler import SimpleTaskHandler

logger = logging.getLogger(f"morpheus.{__name__}")

reset_event = asyncio.Event()


def _build_engine():

    engine = LLMEngine()

    engine.add_node("extracter", node=ExtracterNode())

    async def wait_for_event(values):

        print("Waiting for event...")

        await asyncio.sleep(1)

        print("Waiting for event... Done.")

        return values

    engine.add_node("waiter", inputs=["/extracter"], node=LLMLambdaNode(wait_for_event))

    engine.add_task_handler(inputs=["/extracter"], handler=SimpleTaskHandler())

    return engine


@click.group(name=__name__)
def run():
    pass


@run.command()
@click.option(
    "--num_threads",
    default=os.cpu_count(),
    type=click.IntRange(min=1),
    help="Number of internal pipeline threads to use",
)
@click.option(
    "--pipeline_batch_size",
    default=1024,
    type=click.IntRange(min=1),
    help=("Internal batch size for the pipeline. Can be much larger than the model batch size. "
          "Also used for Kafka consumers"),
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
def pipeline(
    num_threads,
    pipeline_batch_size,
    model_max_batch_size,
):

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.OTHER

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    source_dfs = [cudf.DataFrame({"questions": ["Tell me a story about your best friend.", ]})]

    completion_task = {"task_type": "completion", "task_dict": {"input_keys": ["questions"], }}

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs, repeat=10))

    pipe.add_stage(
        DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(MonitorStage(config, description="Source rate", unit='questions'))

    pipe.add_stage(LLMEngineStage(config, engine=_build_engine()))

    sink = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    pipe.run()

    duration = time.time() - start_time

    print("Got messages: ", sink.get_messages())

    print(f"Total duration: {duration:.2f} seconds")
