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

logger = logging.getLogger(__name__)


def is_valid_service(ctx, param, value):  # pylint: disable=unused-argument
    from morpheus.service.vdb.utils import validate_service
    value = value.lower()
    return validate_service(service_name=value)


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
    default=512,
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
    "--model_name",
    required=True,
    default='all-MiniLM-L6-v2',
    help="The name of the model that is deployed on Triton server",
)
@click.option("--isolate_embeddings",
              is_flag=True,
              default=False,
              help="Whether to fetch all data prior to executing the rest of the pipeline.")
@click.option(
    "--stop_after",
    default=0,
    type=click.IntRange(min=0),
    help="Stop after emitting this many records from the RSS source stage. Useful for testing. Disabled if `0`",
)
@click.option(
    "--enable_cache",
    is_flag=True,
    default=False,
    help="Enable caching of RSS feed request data.",
)
@click.option(
    "--interval_secs",
    default=600,
    type=click.IntRange(min=1),
    help="Interval in seconds between fetching new feed items.",
)
@click.option(
    "--run_indefinitely",
    is_flag=True,
    default=False,
    help=" Indicates whether the process should run continuously.",
)
@click.option(
    "--vector_db_uri",
    type=str,
    default="http://localhost:19530",
    help="URI for connecting to Vector Database server.",
)
@click.option(
    "--vector_db_service",
    type=str,
    default="milvus",
    callback=is_valid_service,
    help="Name of the vector database service to use.",
)
@click.option(
    "--vector_db_resource_name",
    type=str,
    default="RSS",
    help="The identifier of the resource on which operations are to be performed in the vector database.",
)
@click.option(
    "--triton_server_url",
    type=str,
    default="localhost:8001",
    help="Triton server URL.",
)
def pipeline(**kwargs):

    from .pipeline import pipeline as _pipeline

    return _pipeline(**kwargs)


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
