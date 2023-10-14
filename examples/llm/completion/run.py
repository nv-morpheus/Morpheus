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

from ..common.arxiv_source import ArxivSource

logger = logging.getLogger(f"morpheus.{__name__}")


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
@click.option(
    "--model_fea_length",
    default=256,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--embedding_size",
    default=384,
    type=click.IntRange(min=1),
    help="Output size of the embedding model",
)
@click.option(
    "--input_file",
    default="output.csv",
    help="The path to input event stream",
)
@click.option(
    "--output_file",
    default="output.csv",
    help="The path to the file where the inference output will be saved.",
)
@click.option("--server_url", required=True, default='192.168.0.69:8000', help="Tritonserver url")
@click.option(
    "--model_name",
    required=True,
    default='all-mpnet-base-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option("--pre_calc_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
@click.option("--isolate_embeddings",
              is_flag=True,
              default=False,
              help="Whether to pre-calculate the embeddings using Triton")
@click.option("--use_cache",
              type=click.Path(file_okay=True, dir_okay=False),
              default=None,
              help="What cache to use for the confluence documents")
def pipeline(num_threads,
             pipeline_batch_size,
             model_max_batch_size,
             model_fea_length,
             embedding_size,
             input_file,
             output_file,
             server_url,
             model_name,
             pre_calc_embeddings,
             isolate_embeddings,
             use_cache):

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.mode = PipelineModes.NLP
    config.edge_buffer_size = 128

    config.class_labels = [str(i) for i in range(embedding_size)]

    source_dfs = [cudf.DataFrame({"questions": ["Tell me a story about your best friend.", ]})]

    completion_task = {
        "what": "complete",
    }

    pipe = LinearPipeline(config)

    pipe.set_source(InMemorySourceStage(config, dataframes=source_dfs))

    pipe.add_stage(DeserializeStage(config, message_type=ControlMessage, task_type="llm_engine", task_payload=completion_task))

    pipe.add_stage(MonitorStage(config, description="Download rate", unit='pages'))

    if (isolate_embeddings):
        pipe.add_stage(TriggerStage(config))

    if (pre_calc_embeddings):

        # add deserialize stage
        pipe.add_stage(DeserializeStage(config))

        # add preprocessing stage
        pipe.add_stage(
            PreprocessNLPStage(config,
                               vocab_hash_file="data/bert-base-uncased-hash.txt",
                               do_lower_case=True,
                               truncation=True,
                               add_special_tokens=False,
                               column='page_content'))

        pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

        pipe.add_stage(
            TritonInferenceStage(config,
                                 model_name=model_name,
                                 server_url="localhost:8001",
                                 force_convert_inputs=True,
                                 use_shared_memory=True))
        pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name="Arxiv",
                             resource_kwargs=milvus_resource_kwargs,
                             recreate=True,
                             service="milvus",
                             uri="http://localhost:19530"))

    sink = pipe.add_stage(InMemorySinkStage(config, dataframes=source_dfs))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    pipe.run()

    duration = time.time() - start_time

    print(f"Total duration: {duration:.2f} seconds")
