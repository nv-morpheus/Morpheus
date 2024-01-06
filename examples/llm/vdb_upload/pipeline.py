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
import time

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from .module.file_source_pipe import file_source_pipe  # noqa: F401
from .module.rss_source_pipe import rss_source_pipe  # noqa: F401
from .module.schema_transform import schema_transform  # noqa: F401
from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls

logger = logging.getLogger(__name__)


class FileSystemSourceStage:
    def __init__(self, config):
        # Implementation for FileSystemSourceStage
        pass


def setup_rss_source(pipe, config, stop_after, run_indefinitely, enable_cache, interval_secs, model_fea_length,
                     cache_dir="./.cache/http"):
    # TODO(Devin): Read via YAML
    module_config = {
        "module_id": "rss_source_pipe",
        "module_name": "rss_source_pipe",
        "namespace": "morpheus_examples_llm",
        "rss_config": {
            "feed_input": build_rss_urls(),
            "interval_secs": interval_secs,
            "stop_after": stop_after,
            "run_indefinitely": run_indefinitely,
            "batch_size": 128,
            "enable_cache": enable_cache,
            "cache_dir": cache_dir,
            "cooldown_interval": 600,
            "request_timeout": 2.0,
        },
        "web_scraper_config": {
            "chunk_size": model_fea_length,
            "enable_cache": enable_cache,
        },
    }
    sub_pipe = pipe.add_stage(
        LinearModuleSourceStage(config,
                                module_config,
                                output_type=MessageMeta,
                                output_port_name="output"))

    # rss_source = pipe.add_stage(
    #    RSSSourceStage(config,
    #                   feed_input=build_rss_urls(),
    #                   batch_size=128,
    #                   stop_after=stop_after,
    #                   run_indefinitely=run_indefinitely,
    #                   enable_cache=enable_cache,
    #                   interval_secs=interval_secs))

    # monitor_1 = pipe.add_stage(MonitorStage(config, description="RSS Source rate", unit='pages'))
    # web_scraper = pipe.add_stage(WebScraperStage(config, chunk_size=model_fea_length, enable_cache=enable_cache))
    # monitor_2 = pipe.add_stage(MonitorStage(config, description="RSS Download rate", unit='pages'))

    # transform_config = {
    #    "module_id": "schema_transform",
    #    "module_name": "schema_transform_rss",
    #    "namespace": "morpheus_examples_llm",
    #    "schema_transform": {
    #        "summary": {"dtype": "str", "op_type": "select"},
    #        "title": {"dtype": "str", "op_type": "select"},
    #        "content": {"from": "page_content", "dtype": "str", "op_type": "rename"},
    #        "source": {"from": "link", "dtype": "str", "op_type": "rename"}
    #    }
    # }
    # transform = pipe.add_stage(
    #    LinearModulesStage(config,
    #                       transform_config,
    #                       input_type=MessageMeta,
    #                       output_type=MessageMeta,
    #                       input_port_name="input",
    #                       output_port_name="output"))

    ## Connect the pipeline
    # pipe.add_edge(rss_source, monitor_1)
    # pipe.add_edge(monitor_1, web_scraper)
    # pipe.add_edge(web_scraper, monitor_2)
    # pipe.add_edge(monitor_2, transform)

    return sub_pipe


def setup_filesystem_source(pipe, config, filenames, run_indefinitely):
    # TODO(Devin): Read via YAML
    module_config = {
        "module_id": "file_source_pipe",
        "module_name": "file_source_pipe",
        "namespace": "morpheus_examples_llm",
        "file_source_config": {
            "filenames": filenames,
            "watch": run_indefinitely,
        },
    }
    sub_pipe = pipe.add_stage(
        LinearModuleSourceStage(config,
                                module_config,
                                output_type=MessageMeta,
                                output_port_name="output"))

    return sub_pipe


def pipeline(num_threads: int,
             pipeline_batch_size: int,
             model_max_batch_size: int,
             model_fea_length: int,
             embedding_size: int,
             model_name: str,
             isolate_embeddings: bool,
             stop_after: int,
             enable_cache: bool,
             interval_secs: int,
             run_indefinitely: bool,
             vector_db_uri: str,
             vector_db_service: str,
             vector_db_resource_name: str,
             triton_server_url: str,
             source_type: tuple,
             file_source: list):  # New parameter for file sources
    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.edge_buffer_size = 128

    config.class_labels = [str(i) for i in range(embedding_size)]

    pipe = Pipeline(config)

    # Mapping of source types to their setup functions
    source_setup_functions = {
        'rss': lambda: setup_rss_source(pipe, config, stop_after, run_indefinitely, enable_cache, interval_secs,
                                        model_fea_length),
        'filesystem': lambda: setup_filesystem_source(pipe, config, filenames=file_source,
                                                      run_indefinitely=run_indefinitely)
        # Add other source types here in the future
    }

    source_outputs = []
    for src_type in source_type:
        if src_type in source_setup_functions:
            source_output = source_setup_functions[src_type]()
            source_outputs.append(source_output)
        else:
            raise ValueError(f"Unsupported source type: {src_type}")

    deserialize = pipe.add_stage(DeserializeStage(config))

    trigger = None
    if isolate_embeddings:
        trigger = pipe.add_stage(TriggerStage(config))

    nlp_stage = pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file="data/bert-base-uncased-hash.txt",
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='content'))

    monitor_1 = pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

    triton_inference = pipe.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=triton_server_url,
                             force_convert_inputs=True,
                             use_shared_memory=True))
    monitor_2 = pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    vector_db = pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=vector_db_resource_name,
                             resource_kwargs=build_milvus_config(embedding_size=embedding_size),
                             recreate=True,
                             service=vector_db_service,
                             uri=vector_db_uri))

    monitor_3 = pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    # Connect the pipeline
    for source_output in source_outputs:
        pipe.add_edge(source_output, deserialize)

    if (isolate_embeddings):
        pipe.add_edge(deserialize, trigger)
        pipe.add_edge(trigger, nlp_stage)
    else:
        pipe.add_edge(deserialize, nlp_stage)

    pipe.add_edge(nlp_stage, monitor_1)
    pipe.add_edge(monitor_1, triton_inference)
    pipe.add_edge(triton_inference, monitor_2)
    pipe.add_edge(monitor_2, vector_db)
    pipe.add_edge(vector_db, monitor_3)

    start_time = time.time()

    pipe.run()

    return start_time
