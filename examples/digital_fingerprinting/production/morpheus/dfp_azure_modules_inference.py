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
from datetime import datetime

import click
import dfp.modules.dfp_inference_pipeline  # noqa: F401
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.config_generator import ConfigGenerator
from dfp.utils.config_generator import generate_ae_config
from dfp.utils.derive_args import DeriveArgs
from dfp.utils.schema_utils import Schema
from dfp.utils.schema_utils import SchemaBuilder

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage


@click.command()
@click.option(
    "--skip_user",
    multiple=True,
    type=str,
    help="User IDs to skip. Mutually exclusive with only_user",
)
@click.option(
    "--only_user",
    multiple=True,
    type=str,
    help="Only users specified by this option will be included. Mutually exclusive with skip_user",
)
@click.option(
    "--start_time",
    type=click.DateTime(
        formats=['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S%z']),
    default=None,
    help="The start of the time window, if undefined start_date will be `now()-duration`",
)
@click.option(
    "--duration",
    type=str,
    default="1d",
    help="The duration to run starting from start_time",
)
@click.option(
    "--cache_dir",
    type=str,
    default="./.cache/dfp",
    show_envvar=True,
    help="The location to cache data such as S3 downloads and pre-processed data",
)
@click.option("--log_level",
              default=logging.getLevelName(Config().log_level),
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Specify the logging level to use.")
@click.option("--sample_rate_s",
              type=int,
              default=0,
              show_envvar=True,
              help="Minimum time step, in milliseconds, between object logs.")
@click.option(
    "--input_file",
    "-f",
    type=str,
    multiple=True,
    help=("List of files to process. Can specify multiple arguments for multiple files. "
          "Also accepts glob (*) wildcards and schema prefixes such as `s3://`. "
          "For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. "
          "See fsspec documentation for list of possible options."),
)
@click.option('--tracking_uri',
              type=str,
              default="http://mlflow:5000",
              help=("The MLflow tracking URI to connect to the tracking backend."))
def run_pipeline(skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 start_time: datetime,
                 duration,
                 cache_dir,
                 log_level,
                 sample_rate_s,
                 **kwargs):

    derive_args = DeriveArgs(skip_user,
                             only_user,
                             start_time,
                             duration,
                             log_level,
                             cache_dir,
                             sample_rate_s,
                             tracking_uri=kwargs["tracking_uri"],
                             source="azure")

    derive_args.init()

    config: Config = generate_ae_config(labels_file="data/columns_ae_azure.txt",
                                        userid_column_name="username",
                                        timestamp_column_name="timestamp")

    schema_builder = SchemaBuilder(config)
    schema: Schema = schema_builder.build_azure_schema()

    config_generator = ConfigGenerator(config, derive_args, schema)

    module_conf = config_generator.inf_pipe_module_conf()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(MultiFileSource(config, filenames=list(kwargs["input_file"])))

    # Here we add a wrapped module that implements the DFP Inference pipeline
    pipeline.add_stage(LinearModulesStage(config, module_conf, input_port_name="input", output_port_name="output"))

    pipeline.add_stage(MonitorStage(config, description="Inference Pipeline rate", smoothing=0.001))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
