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

import click
import yaml

from morpheus.config import Config
from morpheus.config import PipelineModes

from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls
from .common import build_defualt_milvus_config

logger = logging.getLogger(__name__)


def is_valid_service(ctx, param, value):  # pylint: disable=unused-argument
    """
    Validate the provided vector database service name.

    Checks if the given vector database service name is supported and valid. This is used as a callback function
    for a CLI option to ensure that the user inputs a supported service name.

    Parameters
    ----------
    ctx : click.Context
        The context within which the command is being invoked.
    param : click.Parameter
        The parameter object that this function serves as a callback for.
    value : str
        The value of the parameter to validate.

    Returns
    -------
    str
        The validated and lowercased service name.

    Raises
    ------
    click.BadParameter
        If the provided service name is not supported or invalid.
    """
    from morpheus.service.vdb.utils import validate_service
    value = value.lower()
    return validate_service(service_name=value)


def merge_configs(file_config, cli_config):
    """
    Merge two configuration dictionaries, prioritizing the CLI configuration.

    This function merges configurations provided from a file and the CLI, with the CLI configuration taking precedence
    in case of overlapping keys.

    Parameters
    ----------
    file_config : dict
        The configuration dictionary loaded from a file.
    cli_config : dict
        The configuration dictionary provided via CLI arguments.

    Returns
    -------
    dict
        A merged dictionary with CLI configurations overriding file configurations where they overlap.
    """

    merged_config = file_config.copy()
    merged_config.update({k: v for k, v in cli_config.items() if v is not None})
    return merged_config


def build_cli_configs(source_type,
                      enable_cache,
                      embedding_size,
                      isolate_embeddings,
                      embedding_model_name,
                      enable_monitors,
                      file_source,
                      interval_secs,
                      pipeline_batch_size,
                      run_indefinitely,
                      stop_after,
                      vector_db_resource_name,
                      vector_db_service,
                      vector_db_uri,
                      content_chunking_size,
                      num_threads,
                      rss_request_timeout_sec,
                      model_max_batch_size,
                      model_fea_length,
                      triton_server_url,
                      feed_inputs):
    """
    Create configuration dictionaries based on CLI arguments.

    Constructs individual configuration dictionaries for various components of the data processing pipeline,
    such as source, embeddings, pipeline, tokenizer, and vector database configurations.

    Parameters
    ----------
    source_type : list of str
        Types of data sources (e.g., 'rss', 'filesystem').
    enable_cache : bool
        Flag to enable caching.
    embedding_size : int
        Size of the embeddings.
    isolate_embeddings : bool
        Flag to isolate embeddings.
    embedding_model_name : str
        Name of the embedding model.
    enable_monitors : bool
        Flag to enable monitor functionality.
    file_source : list of str
        File sources or paths to be processed.
    interval_secs : int
        Interval in seconds for operations.
    pipeline_batch_size : int
        Batch size for the pipeline.
    run_indefinitely : bool
        Flag to run the process indefinitely.
    stop_after : int
        Stop after a certain number of records.
    vector_db_resource_name : str
        Name of the resource in the vector database.
    vector_db_service : str
        Name of the vector database service.
    vector_db_uri : str
        URI for the vector database server.
    content_chunking_size : int
        Size of content chunks.
    num_threads : int
        Number of threads to use.
    rss_request_timeout_sec : float
        Timeout in seconds for RSS requests.
    model_max_batch_size : int
        Maximum batch size for the model.
    model_fea_length : int
        Feature length for the model.
    triton_server_url : str
        URL of the Triton server.
    feed_inputs : list of str
        RSS feed inputs.

    Returns
    -------
    tuple
        A tuple containing five dictionaries for source, embeddings, pipeline, tokenizer, and vector database configurations.
    """

    # Source Configuration
    cli_source_conf = {}
    if 'rss' in source_type:
        cli_source_conf['rss'] = {
            'type': 'rss',
            'name': 'rss-cli',
            'config': {
                # RSS feeds can take a while to pull, smaller batch sizes allows the pipeline to feel more responsive
                "batch_size": 32,
                "output_batch_size": 2048,
                "cache_dir": "./.cache/http",
                "cooldown_interval_sec": interval_secs,
                "enable_cache": enable_cache,
                "enable_monitor": enable_monitors,
                "feed_input": feed_inputs if feed_inputs else build_rss_urls(),
                "interval_sec": interval_secs,
                "request_timeout_sec": rss_request_timeout_sec,
                "run_indefinitely": run_indefinitely,
                "stop_after_sec": stop_after,
                "web_scraper_config": {
                    "chunk_size": content_chunking_size,
                    "enable_cache": enable_cache,
                }
            }
        }
    if 'filesystem' in source_type:
        cli_source_conf['filesystem'] = {
            'type': 'filesystem',
            'name': 'filesystem-cli',
            'config': {
                "batch_size": pipeline_batch_size,
                "enable_monitor": enable_monitors,
                "extractor_config": {
                    "chunk_size": content_chunking_size, "num_threads": num_threads
                },
                "filenames": file_source,
                "watch": run_indefinitely
            }
        }

    # Embeddings Configuration
    cli_embeddings_conf = {
        "feature_length": model_fea_length,
        "max_batch_size": model_max_batch_size,
        "model_kwargs": {
            "force_convert_inputs": True,
            "model_name": embedding_model_name,
            "server_url": triton_server_url,
            "use_shared_memory": True,
        },
        "num_threads": num_threads,
    }

    # Pipeline Configuration
    cli_pipeline_conf = {
        "edge_buffer_size": 128,
        "embedding_size": embedding_size,
        "feature_length": model_fea_length,
        "isolate_embeddings": isolate_embeddings,
        "max_batch_size": 256,
        "num_threads": num_threads,
        "pipeline_batch_size": pipeline_batch_size,
    }

    # Tokenizer Configuration
    cli_tokenizer_conf = {
        "model_name": "bert-base-uncased-hash",
        "model_kwargs": {
            "add_special_tokens": False,
            "column": "content",
            "do_lower_case": True,
            "truncation": True,
            "vocab_hash_file": "data/bert-base-uncased-hash.txt",
        }
    }

    # VDB Configuration
    cli_vdb_conf = {
        'embedding_size': embedding_size,
        'recreate': True,
        'resource_kwargs': build_defualt_milvus_config(embedding_size) if (vector_db_service == 'milvus') else None,
        'resource_name': vector_db_resource_name,
        'service': vector_db_service,
        'uri': vector_db_uri,
    }

    return cli_source_conf, cli_embeddings_conf, cli_pipeline_conf, cli_tokenizer_conf, cli_vdb_conf


def build_pipeline_config(pipeline_config: dict):
    """
    Construct a pipeline configuration object from a dictionary.

    Parameters
    ----------
    pipeline_config : dict
        A dictionary containing pipeline configuration parameters.

    Returns
    -------
    Config
        A pipeline configuration object populated with values from the input dictionary.

    Notes
    -----
    This function is responsible for mapping a dictionary of configuration parameters
    into a structured configuration object used by the pipeline.
    """

    config = Config()
    config.mode = PipelineModes.NLP

    embedding_size = pipeline_config.get('embedding_size')

    config.num_threads = pipeline_config.get('num_threads')
    config.pipeline_batch_size = pipeline_config.get('pipeline_batch_size')
    config.model_max_batch_size = pipeline_config.get('max_batch_size')
    config.feature_length = pipeline_config.get('feature_length')
    config.edge_buffer_size = pipeline_config.get('edge_buffer_size')
    config.class_labels = [str(i) for i in range(embedding_size)]

    return config


def build_final_config(vdb_conf_path,
                       cli_source_conf,
                       cli_embeddings_conf,
                       cli_pipeline_conf,
                       cli_tokenizer_conf,
                       cli_vdb_conf):
    """
    Load and merge configurations from the CLI and YAML file.

    This function combines the configurations provided via the CLI with those specified in a YAML file.
    If a YAML configuration file is specified and exists, it will merge its settings with the CLI settings,
    with the YAML settings taking precedence.

    Parameters
    ----------
    vdb_conf_path : str
        Path to the YAML configuration file.
    cli_source_conf : dict
        Source configuration provided via CLI.
    cli_embeddings_conf : dict
        Embeddings configuration provided via CLI.
    cli_pipeline_conf : dict
        Pipeline configuration provided via CLI.
    cli_tokenizer_conf : dict
        Tokenizer configuration provided via CLI.
    cli_vdb_conf : dict
        Vector Database (VDB) configuration provided via CLI.

    Returns
    -------
    dict
        A dictionary containing the final merged configuration for the pipeline, including source, embeddings,
        tokenizer, and VDB configurations.

    Notes
    -----
    The function prioritizes the YAML file configurations over CLI configurations. In case of overlapping
    settings, the values from the YAML file will overwrite those from the CLI.
    """
    final_config = {}

    # Load and merge configurations from the YAML file if it exists
    if vdb_conf_path:
        with open(vdb_conf_path, 'r') as file:
            vdb_pipeline_config = yaml.safe_load(file).get('vdb_pipeline', {})

        embeddings_conf = merge_configs(vdb_pipeline_config.get('embeddings', {}), cli_embeddings_conf)
        pipeline_conf = merge_configs(vdb_pipeline_config.get('pipeline', {}), cli_pipeline_conf)
        source_conf = vdb_pipeline_config.get('sources', []) + list(cli_source_conf.values())
        tokenizer_conf = merge_configs(vdb_pipeline_config.get('tokenizer', {}), cli_tokenizer_conf)
        vdb_conf = vdb_pipeline_config.get('vdb', {})
        resource_schema = vdb_conf.pop("resource_shema", None)

        if resource_schema:
            vdb_conf["resource_kwargs"] = build_milvus_config(resource_schema)
        vdb_conf = merge_configs(vdb_conf, cli_vdb_conf)

        # TODO: class labels depends on this, so it should be a pipeline level parameter, not a vdb level parameter
        pipeline_conf['embedding_size'] = vdb_conf.get('embedding_size', 384)

        final_config.update({
            'embeddings_config': embeddings_conf,
            'pipeline_config': build_pipeline_config(pipeline_conf),
            'source_config': source_conf,
            'tokenizer_config': tokenizer_conf,
            'vdb_config': vdb_conf,
        })
    else:
        # Use CLI configurations only
        final_config.update({
            'embeddings_config': cli_embeddings_conf,
            'pipeline_config': build_pipeline_config(cli_pipeline_conf),
            'source_config': list(cli_source_conf.values()),
            'tokenizer_config': cli_tokenizer_conf,
            'vdb_config': cli_vdb_conf,
        })

    return final_config


@click.group(name=__name__)
def run():
    pass


@run.command()
@click.option(
    "--content_chunking_size",
    default=512,  # Set a sensible default value
    type=click.IntRange(min=1),  # Ensure that only positive integers are valid
    help="The size of content chunks for processing.")
@click.option(
    "--embedding_size",
    default=384,
    type=click.IntRange(min=1),
    help="Output size of the embedding model",
)
@click.option(
    "--enable_cache",
    is_flag=True,
    default=False,
    help="Enable caching of RSS feed request data.",
)
@click.option("--enable_monitors", is_flag=True, default=False, help="Enable or disable monitor functionality.")
@click.option('--file_source', multiple=True, default=[], type=str, help='List of file sources/paths to be processed.')
@click.option('--feed_inputs', multiple=True, default=[], type=str, help='List of RSS source feeds to process.')
@click.option(
    "--interval_secs",
    default=600,
    type=click.IntRange(min=1),
    help="Interval in seconds between fetching new feed items.",
)
@click.option("--isolate_embeddings",
              is_flag=True,
              default=False,
              help="Whether to fetch all data prior to executing the rest of the pipeline.")
@click.option(
    "--model_fea_length",
    default=512,
    type=click.IntRange(min=1),
    help="Features length to use for the model",
)
@click.option(
    "--model_max_batch_size",
    default=64,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--embedding_model_name",
    required=True,
    default='all-MiniLM-L6-v2',
    help="The name of the model that is deployed on Triton server",
)
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
    "--run_indefinitely",
    is_flag=True,
    default=False,
    help="Indicates whether the process should run continuously.",
)
@click.option(
    "--rss_request_timeout_sec",
    default=2.0,  # Set a default value, adjust as needed
    type=click.FloatRange(min=0.0),  # Ensure that only non-negative floats are valid
    help="Timeout in seconds for RSS feed requests.")
@click.option("--source_type",
              multiple=True,
              type=click.Choice(['rss', 'filesystem'], case_sensitive=False),
              default=[],
              show_default=True,
              help="The type of source to use. Can specify multiple times for different source types.")
@click.option(
    "--stop_after",
    default=0,
    type=click.IntRange(min=0),
    help="Stop after emitting this many records from the RSS source stage. Useful for testing. Disabled if `0`",
)
@click.option(
    "--triton_server_url",
    type=str,
    default="localhost:8001",
    help="Triton server URL.",
)
@click.option(
    "--vdb_config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a YAML configuration file.",
)
@click.option(
    "--vector_db_resource_name",
    type=str,
    default="VDBUploadExample",
    help="The identifier of the resource on which operations are to be performed in the vector database.",
)
@click.option(
    "--vector_db_service",
    type=str,
    default="milvus",
    callback=is_valid_service,
    help="Name of the vector database service to use.",
)
@click.option(
    "--vector_db_uri",
    type=str,
    default="http://localhost:19530",
    help="URI for connecting to Vector Database server.",
)
def pipeline(**kwargs):
    """
    Configure and run the data processing pipeline based on the specified command-line options.

    This function initializes and runs the data processing pipeline using configurations provided
    via command-line options. It supports customization for various components of the pipeline such as
    source type, embedding model, and vector database parameters.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing command-line options.

    Returns
    -------
    The result of the internal pipeline function call.
    """
    vdb_config_path = kwargs.pop('vdb_config_path', None)
    cli_source_conf, cli_embed_conf, cli_pipe_conf, cli_tok_conf, cli_vdb_conf = build_cli_configs(**kwargs)
    final_config = build_final_config(vdb_config_path,
                                      cli_source_conf,
                                      cli_embed_conf,
                                      cli_pipe_conf,
                                      cli_tok_conf,
                                      cli_vdb_conf)

    # Call the internal pipeline function with the final config dictionary
    from .pipeline import pipeline as _pipeline
    return _pipeline(**final_config)


@run.command()
@click.option(
    "--model_name",
    required=True,
    default='all-MiniLM-L6-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--save_cache",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Location to save the cache to",
)
def langchain(**kwargs):
    from .langchain import chain

    return chain(**kwargs)


@run.command()
@click.option(
    "--model_name",
    required=True,
    default='all-MiniLM-L6-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option(
    "--model_seq_length",
    default=512,
    type=click.IntRange(min=1),
    help="Accepted input size of the text tokens",
)
@click.option(
    "--max_batch_size",
    default=256,
    type=click.IntRange(min=1),
    help="Max batch size for the model config",
)
@click.option(
    "--triton_repo",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory of the Triton Model Repo where the model will be saved",
)
@click.option(
    "--output_model_name",
    default=None,
    help="Overrides the model name that is used in triton. Defaults to `model_name`",
)
def export_triton_model(**kwargs):
    from .export_model import build_triton_model

    return build_triton_model(**kwargs)
