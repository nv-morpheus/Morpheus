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
import typing

import pymilvus

from morpheus.config import Config
from morpheus.messages.multi_message import MultiMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from .module.file_source_pipe import FileSourcePipe
from .module.rss_source_pipe import RSSSourcePipe

logger = logging.getLogger(__name__)


def validate_source_config(source_info: typing.Dict[str, any]) -> None:
    """
    Validates the configuration of a source.

    This function checks whether the given source configuration dictionary
    contains all required keys: 'type', 'name', and 'config'.

    Parameters
    ----------
    source_info : typing.Dict[str, any]
        The source configuration dictionary to validate.

    Raises
    ------
    ValueError
        If any of the required keys ('type', 'name', 'config') are missing
        in the source configuration.
    """
    if ('type' not in source_info or 'name' not in source_info or 'config' not in source_info):
        raise ValueError(f"Each source must have 'type', 'name', and 'config':\n {source_info}")


def setup_rss_source(pipe: Pipeline, config: Config, source_name: str, rss_config: typing.Dict[str, typing.Any]):
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
    rss_config : typing.Dict[str, Any]
        Configuration parameters for the RSS source stage.

    Returns
    -------
    SubPipeline
        The sub-pipeline stage created for the RSS source.
    """
    module_definition = RSSSourcePipe.get_definition(module_name=f"rss_source_pipe__{source_name}",
                                                     module_config={"rss_config": rss_config}, )
    rss_pipe = pipe.add_stage(
        LinearModuleSourceStage(config,
                                module_definition,
                                output_type=MultiMessage,
                                output_port_name="output"))

    return rss_pipe


def setup_filesystem_source(pipe: Pipeline, config: Config, source_name: str, fs_config: typing.Dict[str, typing.Any]):
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
    fs_config : typing.Dict[str, Any]
        Configuration parameters for the filesystem source stage.

    Returns
    -------
    SubPipeline
        The sub-pipeline stage created for the filesystem source.
    """

    module_definition = FileSourcePipe.get_definition(module_name=f"file_source_pipe__{source_name}",
                                                      module_config={"file_source_config": fs_config})
    file_pipe = pipe.add_stage(
        LinearModuleSourceStage(config,
                                module_definition,
                                output_type=MultiMessage,
                                output_port_name="output"))

    return file_pipe


def setup_custom_source(pipe: Pipeline, config: Config, source_name: str, custom_config: typing.Dict[str, typing.Any]):
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
    custom_config : typing.Dict[str, Any]
        Configuration parameters for the custom source stage, including
        the module_id, module_name, namespace, and any additional parameters.

    Returns
    -------
    SubPipeline
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


def process_vdb_sources(pipe: Pipeline, config: Config, vdb_source_config: typing.List[typing.Dict]) -> typing.List:
    """
    Processes and sets up sources defined in a vdb_source_config.

    This function reads the source configurations provided in vdb_source_config and
    sets up each source based on its type ('rss', 'filesystem', or 'custom').
    It validates each source configuration and then calls the appropriate setup
    function to add the source to the pipeline.

    Parameters
    ----------
    pipe : Pipeline
        The pipeline to which the sources will be added.
    config : Config
        Configuration object for the pipeline.
    vdb_source_config : List[Dict]
        A list of dictionaries, each containing the configuration for a source.

    Returns
    -------
    list
        A list of the sub-pipeline stages created for each defined source.

    Raises
    ------
    ValueError
        If an unsupported source type is encountered in the configuration.
    """
    vdb_sources = []
    for source_info in vdb_source_config:
        validate_source_config(source_info)  # Assuming this function exists
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

    return vdb_sources


def build_milvus_config(embedding_size: int) -> Dict[str, Any]:
    """
    Builds the configuration for Milvus.

    This function creates a dictionary configuration for a Milvus collection.
    It includes the index configuration and the schema configuration, with
    various fields like id, title, link, summary, page_content, and embedding.

    Parameters
    ----------
    embedding_size : int
        The size of the embedding vector.

    Returns
    -------
    typing.Dict[str, Any]
        A dictionary containing the configuration settings for Milvus.
    """

    milvus_resource_kwargs = {
        "index_conf": {
            "field_name": "embedding",
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 8,
                "efConstruction": 64,
            },
        },
        "schema_conf": {
            "enable_dynamic_field": True,
            "schema_fields": [
                pymilvus.FieldSchema(name="id",
                                     dtype=pymilvus.DataType.INT64,
                                     description="Primary key for the collection",
                                     is_primary=True,
                                     auto_id=True).to_dict(),
                pymilvus.FieldSchema(name="title",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The title of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="link",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The URL of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="summary",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="The summary of the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="page_content",
                                     dtype=pymilvus.DataType.VARCHAR,
                                     description="A chunk of text from the RSS Page",
                                     max_length=65_535).to_dict(),
                pymilvus.FieldSchema(name="embedding",
                                     dtype=pymilvus.DataType.FLOAT_VECTOR,
                                     description="Embedding vectors",
                                     dim=embedding_size).to_dict(),
            ],
            "description": "Test collection schema"
        }
    }

    return milvus_resource_kwargs


def build_rss_urls() -> typing.List[str]:
    """
    Builds a list of RSS feed URLs.

    Returns
    -------
    typing.List[str]
        A list of URLs as strings, each pointing to a different RSS feed.
    """

    return [
        "https://www.theregister.com/security/headlines.atom",
        "https://isc.sans.edu/dailypodcast.xml",
        "https://threatpost.com/feed/",
        "http://feeds.feedburner.com/TheHackersNews?format=xml",
        "https://www.bleepingcomputer.com/feed/",
        "https://therecord.media/feed/",
        "https://blog.badsectorlabs.com/feeds/all.atom.xml",
        "https://krebsonsecurity.com/feed/",
        "https://www.darkreading.com/rss_simple.asp",
        "https://blog.malwarebytes.com/feed/",
        "https://msrc.microsoft.com/blog/feed",
        "https://securelist.com/feed",
        "https://www.crowdstrike.com/blog/feed/",
        "https://threatconnect.com/blog/rss/",
        "https://news.sophos.com/en-us/feed/",
        "https://www.us-cert.gov/ncas/current-activity.xml",
        "https://www.csoonline.com/feed",
        "https://www.cyberscoop.com/feed",
        "https://research.checkpoint.com/feed",
        "https://feeds.fortinet.com/fortinet/blog/threat-research",
        "https://www.mcafee.com/blogs/rss",
        "https://www.digitalshadows.com/blog-and-research/rss.xml",
        "https://www.nist.gov/news-events/cybersecurity/rss.xml",
        "https://www.sentinelone.com/blog/rss/",
        "https://www.bitdefender.com/blog/api/rss/labs/",
        "https://www.welivesecurity.com/feed/",
        "https://unit42.paloaltonetworks.com/feed/",
        "https://mandiant.com/resources/blog/rss.xml",
        "https://www.wired.com/feed/category/security/latest/rss",
        "https://www.wired.com/feed/tag/ai/latest/rss",
        "https://blog.google/threat-analysis-group/rss/",
        "https://intezer.com/feed/",
    ]
