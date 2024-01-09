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

import yaml

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.multi_message import MultiMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.output.write_to_vector_db_stage import WriteToVectorDBStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage

from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls
from .module.file_source_pipe import FileSourcePipe
from .module.rss_source_pipe import RSSSourcePipe
from .module.schema_transform import schema_transform  # noqa: F401

logger = logging.getLogger(__name__)


def setup_rss_source(pipe, config, source_name, rss_config):
    """
    Set up the RSS source stage in the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        The pipeline to which the RSS source stage will be added.
    config : Config
        Configuration object for the pipeline.
    source_name : str
        The name of the RSS source stage.
    rss_config : dict
        Configuration parameters for the RSS source stage.

    Returns
    -------
    sub_pipe
        The sub-pipeline stage created for the RSS source.
    """
    module_definition = RSSSourcePipe.get_definition(
        module_name=f"rss_source_pipe__{source_name}",
        module_config={"rss_config": rss_config},
    )
    rss_pipe = pipe.add_stage(
        LinearModuleSourceStage(config, module_definition, output_type=MultiMessage, output_port_name="output"))

    return rss_pipe


def setup_filesystem_source(pipe, config, source_name, fs_config):
    """
    Set up the filesystem source stage in the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        The pipeline to which the filesystem source stage will be added.
    config : Config
        Configuration object for the pipeline.
    source_name : str
        The name of the filesystem source stage.
    fs_config : dict
        Configuration parameters for the filesystem source stage.

    Returns
    -------
    sub_pipe
        The sub-pipeline stage created for the filesystem source.
    """
    module_definition = FileSourcePipe.get_definition(module_name=f"file_source_pipe__{source_name}",
                                                      module_config={"file_source_config": fs_config})
    file_pipe = pipe.add_stage(
        LinearModuleSourceStage(config, module_definition, output_type=MultiMessage, output_port_name="output"))

    return file_pipe


def setup_custom_source(pipe, config, source_name, custom_config):
    """
    Setup a custom source stage in the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        The pipeline to which the custom source stage will be added.
    config : Config
        Configuration object for the pipeline.
    source_name : str
        The name of the custom source stage.
    custom_config : dict
        Configuration parameters for the custom source stage, including
        the module_id, module_name, namespace, and any additional parameters.

    Returns
    -------
    sub_pipe
        The sub-pipeline stage created for the custom source.
    """
    module_config = {
        "module_id": custom_config['module_id'],
        "module_name": f"{custom_config['module_id']}__{source_name}",
        "namespace": custom_config['namespace'],
    }

    if ('config_name_mapping' in custom_config):
        module_config[custom_config['config_name_mapping']] = custom_config
    else:
        module_config['config'] = custom_config

    # Adding the custom module stage to the pipeline
    custom_pipe = pipe.add_stage(
        LinearModuleSourceStage(config,
                                module_config,
                                output_type=MultiMessage,
                                output_port_name=custom_config.get('module_output_id', 'output')))

    return custom_pipe


def validate_source_config(source_info):
    """
    Validates the configuration of a source.

    This function checks whether the given source configuration dictionary
    contains all required keys: 'type', 'name', and 'config'.

    Parameters
    ----------
    source_info : dict
        The source configuration dictionary to validate.

    Raises
    ------
    ValueError
        If any of the required keys ('type', 'name', 'config') are missing
        in the source configuration.
    """
    if ('type' not in source_info or 'name' not in source_info or 'config' not in source_info):
        raise ValueError(f"Each source must have 'type', 'name', and 'config':\n {source_info}")


def process_vdb_sources(pipe, config, vdb_config_path):
    """
    Processes and sets up sources defined in a YAML configuration file.

    This function reads the given YAML file and sets up each source
    defined within it, based on its type ('rss', 'filesystem', or 'custom').
    It validates each source configuration and then calls the appropriate
    setup function to add the source to the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        The pipeline to which the sources will be added.
    config : Config
        Configuration object for the pipeline.
    vdb_config_path : str
        Path to the YAML file containing the source configurations.

    Returns
    -------
    list
        A list of the sub-pipeline stages created for each defined source.

    Raises
    ------
    ValueError
        If an unsupported source type is encountered in the configuration.
    """
    with open(vdb_config_path, 'r') as file:
        vdb_pipeline = yaml.safe_load(file).get('vdb_pipeline', {})
        vdb_source_config = vdb_pipeline.get('sources', [])
        vdb_database = vdb_pipeline.get('vdb', {})
        vdb_embeddings = vdb_pipeline.get('embeddings', {})

    vdb_sources = []
    for source_info in vdb_source_config:
        validate_source_config(source_info)
        source_type = source_info['type']
        source_name = source_info['name']
        source_config = source_info['config']

        if (source_type == 'rss'):
            vdb_sources.append(setup_rss_source(pipe, config, source_name, source_config))
        elif (source_type == 'filesystem'):
            vdb_sources.append(setup_filesystem_source(pipe, config, source_name, source_config))
        elif (source_type == 'custom'):
            vdb_sources.append(setup_custom_source(pipe, config, source_name, source_config))
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    return vdb_sources, vdb_database, vdb_embeddings


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
             file_source: list,
             source_type: tuple,
             vdb_config: str):  # New parameter for file sources
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

    # TODO(Devin) Merge with priority against cli parameters
    vdb_sources = []
    vdb_database = {}
    vdb_embeddings = {}
    if (vdb_config):
        vdb_sources, vdb_database, vdb_embeddings = process_vdb_sources(pipe, config, vdb_config)

    # Additional source setup using command-line options if needed
    source_setup_functions = {
        'rss':
            lambda: setup_rss_source(
                pipe,
                config,
                "cli_rss_source",
                {
                    "batch_size": 128,  # Example value
                    "cache_dir": "./.cache/http",
                    "cooldown_interval": 600,
                    "enable_cache": enable_cache,
                    "feed_input": build_rss_urls(),
                    "interval_secs": interval_secs,
                    "request_timeout": 2.0,
                    "run_indefinitely": run_indefinitely,
                    "stop_after": stop_after,
                }),
        'filesystem':
            lambda: setup_filesystem_source(
                pipe, config, "cli_filesystem_source", {
                    "filenames": file_source, "watch": run_indefinitely
                })
        # Add other source types here in the future
    }

    for src_type in source_type:
        if src_type in source_setup_functions:
            source_output = source_setup_functions[src_type]()
            vdb_sources.append(source_output)
        else:
            raise ValueError(f"Unsupported source type: {src_type}")

    trigger = None
    if (isolate_embeddings):
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
                            resource_name=vdb_database.get('resource_name', vector_db_resource_name),
                            resource_kwargs=build_milvus_config(embedding_size=vdb_embeddings.get('size', 384)),
                            recreate=vdb_database.get('recreate', True),
                            service=vdb_database.get('service', vector_db_service),
                            uri=vdb_database.get('uri', vector_db_uri)))

    monitor_3 = pipe.add_stage(MonitorStage(config, description="Upload rate", unit="events", delayed_start=True))

    # Connect the pipeline
    for source_output in vdb_sources:
        if (isolate_embeddings):
            pass
            pipe.add_edge(source_output, trigger)
        else:
            pipe.add_edge(source_output, nlp_stage)

    if (isolate_embeddings):
        pipe.add_edge(trigger, nlp_stage)

    pipe.add_edge(nlp_stage, monitor_1)
    pipe.add_edge(monitor_1, triton_inference)
    pipe.add_edge(triton_inference, monitor_2)
    pipe.add_edge(monitor_2, vector_db)
    pipe.add_edge(vector_db, monitor_3)

    start_time = time.time()

    pipe.run()

    return start_time
