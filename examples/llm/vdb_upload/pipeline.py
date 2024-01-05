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
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from .module.transform_module import schema_transform  # noqa: F401
from .stages.multi_file_source import MultiFileSource
from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls
from ..common.web_scraper_stage import WebScraperStage

logger = logging.getLogger(__name__)


class FileSystemSourceStage:
    def __init__(self, config):
        # Implementation for FileSystemSourceStage
        pass


def setup_rss_source(pipe, config, stop_after, run_indefinitely, enable_cache, interval_secs, model_fea_length):
    pipe.set_source(
        RSSSourceStage(config,
                       feed_input=build_rss_urls(),
                       batch_size=128,
                       stop_after=stop_after,
                       run_indefinitely=run_indefinitely,
                       enable_cache=enable_cache,
                       interval_secs=interval_secs))

    pipe.add_stage(MonitorStage(config, description="RSS Source rate", unit='pages'))
    pipe.add_stage(WebScraperStage(config, chunk_size=model_fea_length, enable_cache=enable_cache))
    pipe.add_stage(MonitorStage(config, description="RSS Download rate", unit='pages'))

    transform_config = {
        "module_id": "schema_transform",
        "module_name": "schema_transform_rss",
        "namespace": "morpheus_examples_llm",
        "schema_transform": {
            "summary": {"dtype": "str", "op_type": "select"},
            "title": {"dtype": "str", "op_type": "select"},
            "content": {"from": "page_content", "dtype": "str", "op_type": "rename"},
            "source": {"from": "link", "dtype": "str", "op_type": "rename"}
        }
    }
    pipe.add_stage(
        LinearModulesStage(config,
                           transform_config,
                           input_type=MessageMeta,
                           output_type=MessageMeta,
                           input_port_name="input",
                           output_port_name="output"))


def setup_filesystem_source(pipe, config, filenames, run_indefinitely):
    # Initialize the MultiFileSource stage with the run_indefinitely parameter for watch
    file_source_stage = MultiFileSource(
        config,
        filenames=filenames,
        watch=run_indefinitely
    )

    pipe.set_source(file_source_stage)
    # Add any additional stages specific to filesystem processing if needed
    pipe.add_stage(MonitorStage(config, description="Filesystem Source rate", unit='files'))

    # TODO(Devin)
    # Need a stage to process the file contents into a dataframe

    # TODO(Devin)
    # Add a schema_transform to ensure the dataframe has the expected schema
    transform_config = {
        "module_id": "schema_transform",
        "module_name": "schema_transform_rss",
        "namespace": "morpheus_examples_llm",
        "schema_transform": {
            "summary": {"dtype": "str", "op_type": "select"},
            "title": {"dtype": "str", "op_type": "select"},
            "content": {"dtype": "str", "op_type": "select"},
            "source": {"dtype": "str", "op_type": "select"}
        }
    }
    pipe.add_stage(
        LinearModulesStage(config,
                           transform_config,
                           input_type=MessageMeta,
                           output_type=MessageMeta,
                           input_port_name="input",
                           output_port_name="output"))


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
             source_type: tuple):
    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size
    config.feature_length = model_fea_length
    config.edge_buffer_size = 128

    config.class_labels = [str(i) for i in range(embedding_size)]

    pipe = LinearPipeline(config)

    if ('rss' in source_type):
        setup_rss_source(pipe, config, stop_after, run_indefinitely, enable_cache, interval_secs, model_fea_length)
    elif ('filesystem' in source_type):
        setup_filesystem_source(pipe, config)
    else:
        raise ValueError("Unsupported source type")

    pipe.add_stage(DeserializeStage(config))

    if isolate_embeddings:
        pipe.add_stage(TriggerStage(config))

    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file="data/bert-base-uncased-hash.txt",
                           do_lower_case=True,
                           truncation=True,
                           add_special_tokens=False,
                           column='content'))

    pipe.add_stage(MonitorStage(config, description="Tokenize rate", unit='events', delayed_start=True))

    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name=model_name,
                             server_url=triton_server_url,
                             force_convert_inputs=True,
                             use_shared_memory=True))
    pipe.add_stage(MonitorStage(config, description="Inference rate", unit="events", delayed_start=True))

    pipe.add_stage(
        WriteToVectorDBStage(config,
                             resource_name=vector_db_resource_name,
                             resource_kwargs=build_milvus_config(embedding_size=embedding_size),
                             recreate=True,
                             service=vector_db_service,
                             uri=vector_db_uri))

    pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    start_time = time.time()

    pipe.run()

    return start_time
