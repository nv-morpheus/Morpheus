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

from ..common.utils import build_milvus_config
from ..common.utils import build_rss_urls

logger = logging.getLogger(__name__)


def is_valid_service(ctx, param, value):  # pylint: disable=unused-argument
    from morpheus.service.vdb.utils import validate_service
    value = value.lower()
    return validate_service(service_name=value)


def merge_configs(file_config, cli_config):
    merged_config = file_config.copy()
    merged_config.update({k: v for k, v in cli_config.items() if v is not None})
    return merged_config


def build_cli_configs(source_type, enable_cache, embedding_size, isolate_embeddings, embedding_model_name,
                      enable_monitors, file_source, interval_secs, pipeline_batch_size, run_indefinitely, stop_after,
                      vector_db_resource_name, vector_db_service, vector_db_uri, content_chunking_size, num_threads,
                      rss_request_timeout_sec, model_max_batch_size, model_fea_length, triton_server_url, feed_inputs):
    """
    Create configuration dictionaries for CLI arguments.

    Returns
    -------
    Tuple containing configurations for source, embeddings, pipeline, tokenizer, and vdb.
    """
    # Source Configuration
    cli_source_config = {}
    if 'rss' in source_type:
        cli_source_config['rss'] = {
            'type': 'rss',
            'name': 'rss-cli',
            'config': {
                "batch_size": pipeline_batch_size,
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
        cli_source_config['filesystem'] = {
            'type': 'filesystem',
            'name': 'filesystem-cli',
            'config': {
                "batch_size": pipeline_batch_size,
                "enable_monitor": enable_monitors,
                "extractor_config": {
                    "chunk_size": content_chunking_size,
                    "num_threads": num_threads
                },
                "filenames": file_source,
                "watch": run_indefinitely
            }
        }

    # Embeddings Configuration
    cli_embeddings_config = {
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
    cli_pipeline_config = {
        "feature_length": model_fea_length,
        "isolate_embeddings": isolate_embeddings,
        "num_threads": num_threads,
        "pipeline_batch_size": pipeline_batch_size,
    }

    # Tokenizer Configuration
    cli_tokenizer_config = {
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
    cli_vdb_config = {
        'embedding_size': embedding_size,
        'recreate': True,
        'resource_kwargs': build_milvus_config(embedding_size) if (vector_db_service == 'milvus') else None,
        'resource_name': vector_db_resource_name,
        'service': vector_db_service,
        'uri': vector_db_uri,
    }

    return cli_source_config, cli_embeddings_config, cli_pipeline_config, cli_tokenizer_config, cli_vdb_config


def build_final_config(vdb_config_path, cli_source_config, cli_embeddings_config, cli_pipeline_config,
                       cli_tokenizer_config,
                       cli_vdb_config):
    """
    Load and merge configurations from the CLI and YAML file.
    """
    final_config = {}

    # Load and merge configurations from the YAML file if it exists
    if vdb_config_path:
        with open(vdb_config_path, 'r') as file:
            vdb_pipeline_config = yaml.safe_load(file).get('vdb_pipeline', {})

        embeddings_config = merge_configs(vdb_pipeline_config.get('embeddings', {}), cli_embeddings_config)
        pipeline_config = merge_configs(vdb_pipeline_config.get('pipeline', {}), cli_pipeline_config)
        tokenizer_config = merge_configs(vdb_pipeline_config.get('tokenizer', {}), cli_tokenizer_config)
        vdb_config = merge_configs(vdb_pipeline_config.get('vdb', {}), cli_vdb_config)

        # Append CLI source configurations
        source_config = vdb_pipeline_config.get('sources', []) + list(cli_source_config.values())

        final_config.update({
            'embeddings_config': embeddings_config,
            'pipeline_config': pipeline_config,
            'source_config': source_config,
            'tokenizer_config': tokenizer_config,
            'vdb_config': vdb_config,
        })
    else:
        # Use CLI configurations only
        final_config.update({
            'embeddings_config': cli_embeddings_config,
            'pipeline_config': cli_pipeline_config,
            'source_config': list(cli_source_config.values()),
            'tokenizer_config': cli_tokenizer_config,
            'vdb_config': cli_vdb_config,
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
    help="The size of content chunks for processing."
)
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
@click.option(
    "--enable_monitors",
    is_flag=True,
    default=False,
    help="Enable or disable monitor functionality."
)
@click.option(
    '--file_source',
    multiple=True,
    default=[],
    type=str,
    help='List of file sources/paths to be processed.')
@click.option(
    '--feed_inputs',
    multiple=True,
    default=[],
    type=str,
    help='List of RSS source feeds to process.')
@click.option(
    "--interval_secs",
    default=600,
    type=click.IntRange(min=1),
    help="Interval in seconds between fetching new feed items.",
)
@click.option(
    "--isolate_embeddings",
    is_flag=True,
    default=False,
    help="Whether to fetch all data prior to executing the rest of the pipeline."
)
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
    help="Timeout in seconds for RSS feed requests."
)
@click.option(
    "--source_type",
    multiple=True,
    type=click.Choice(['rss', 'filesystem'], case_sensitive=False),
    default=[],
    show_default=True,
    help="The type of source to use. Can specify multiple times for different source types."
)
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
    default="RSS",
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
    Main pipeline function to configure and run the data processing pipeline.
    """
    vdb_config_path = kwargs.pop('vdb_config_path', None)
    cli_source_conf, cli_embed_conf, cli_pipe_conf, cli_tok_conf, cli_vdb_conf = build_cli_configs(**kwargs)
    final_config = build_final_config(vdb_config_path, cli_source_conf, cli_embed_conf, cli_pipe_conf, cli_tok_conf,
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
