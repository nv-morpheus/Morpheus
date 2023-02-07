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
from functools import partial

import click
import dfp.modules.dfp_inference_pipeline  # noqa: F401
import dfp.modules.dfp_postprocessing  # noqa: F401
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.derive_args import DeriveArgs
from dfp.utils.derive_args import get_ae_config
from dfp.utils.derive_args import pyobj2str
from dfp.utils.module_ids import DFP_DATA_PREP
from dfp.utils.module_ids import DFP_INFERENCE
from dfp.utils.module_ids import DFP_INFERENCE_PIPELINE
from dfp.utils.module_ids import DFP_POST_PROCESSING
from dfp.utils.module_ids import DFP_ROLLING_WINDOW
from dfp.utils.module_ids import DFP_SPLIT_USERS
from dfp.utils.regex_utils import iso_date_regex_pattern

from morpheus._lib.common import FilterSource
from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.messages.multi_message import MultiMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import create_increment_col
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import MODULE_NAMESPACE


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

    da = DeriveArgs(skip_user,
                    only_user,
                    start_time,
                    duration,
                    log_level,
                    cache_dir,
                    tracking_uri=kwargs["tracking_uri"],
                    source="azure")

    da.init()

    config: Config = get_ae_config(labels_file="data/columns_ae_azure.txt",
                                   userid_column_name="username",
                                   timestamp_column_name="timestamp")

    # Specify the column names to ensure all data is uniform
    source_column_info = [
        DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="time"),
        RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="properties.userPrincipalName"),
        RenameColumn(name="appDisplayName", dtype=str, input_name="properties.appDisplayName"),
        ColumnInfo(name="category", dtype=str),
        RenameColumn(name="clientAppUsed", dtype=str, input_name="properties.clientAppUsed"),
        RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="properties.deviceDetail.browser"),
        RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="properties.deviceDetail.displayName"),
        RenameColumn(name="deviceDetailoperatingSystem",
                     dtype=str,
                     input_name="properties.deviceDetail.operatingSystem"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "properties.location.city",
                            "properties.location.countryOrRegion",
                        ],
                        sep=", "),
        RenameColumn(name="statusfailureReason", dtype=str, input_name="properties.status.failureReason"),
    ]

    source_schema = DataFrameInputSchema(json_columns=["properties"], column_info=source_column_info)

    # Preprocessing schema
    preprocess_column_info = [
        ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
        ColumnInfo(name=config.ae.userid_column_name, dtype=str),
        ColumnInfo(name="appDisplayName", dtype=str),
        ColumnInfo(name="clientAppUsed", dtype=str),
        ColumnInfo(name="deviceDetailbrowser", dtype=str),
        ColumnInfo(name="deviceDetaildisplayName", dtype=str),
        ColumnInfo(name="deviceDetailoperatingSystem", dtype=str),
        ColumnInfo(name="statusfailureReason", dtype=str),

        # Derived columns
        IncrementColumn(name="logcount",
                        dtype=int,
                        input_name=config.ae.timestamp_column_name,
                        groupby_column=config.ae.userid_column_name),
        CustomColumn(name="locincrement",
                     dtype=int,
                     process_column_fn=partial(create_increment_col, column_name="location")),
        CustomColumn(name="appincrement",
                     dtype=int,
                     process_column_fn=partial(create_increment_col, column_name="appDisplayName")),
    ]

    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    encoding = "latin1"

    # Convert schema as a string
    source_schema_str = pyobj2str(source_schema, encoding=encoding)
    preprocess_schema_str = pyobj2str(preprocess_schema, encoding=encoding)

    module_config = {
        "module_id": DFP_INFERENCE_PIPELINE,
        "module_name": "dfp_inference_pipeline",
        "namespace": MODULE_NAMESPACE,
        FILE_BATCHER: {
            "module_id": FILE_BATCHER,
            "module_name": "file_batcher",
            "namespace": MODULE_NAMESPACE,
            "period": "D",
            "sampling_rate_s": sample_rate_s,
            "start_time": da.start_time,
            "end_time": da.end_time,
            "iso_date_regex_pattern": iso_date_regex_pattern
        },
        FILE_TO_DF: {
            "module_id": FILE_TO_DF,
            "module_name": "FILE_TO_DF",
            "namespace": MODULE_NAMESPACE,
            "timestamp_column_name": config.ae.timestamp_column_name,
            "parser_kwargs": {
                "lines": False, "orient": "records"
            },
            "cache_dir": cache_dir,
            "filter_null": True,
            "file_type": "JSON",
            "schema": {
                "schema_str": source_schema_str, "encoding": encoding
            }
        },
        DFP_SPLIT_USERS: {
            "module_id": DFP_SPLIT_USERS,
            "module_name": "dfp_split_users",
            "namespace": MODULE_NAMESPACE,
            "include_generic": da.include_generic,
            "include_individual": da.include_individual,
            "skip_users": da.skip_users,
            "only_users": da.only_users,
            "timestamp_column_name": config.ae.timestamp_column_name,
            "userid_column_name": config.ae.userid_column_name,
            "fallback_username": config.ae.fallback_username
        },
        DFP_ROLLING_WINDOW: {
            "module_id": DFP_ROLLING_WINDOW,
            "module_name": "dfp_rolling_window",
            "namespace": MODULE_NAMESPACE,
            "min_history": 1,
            "min_increment": 0,
            "max_history": duration,
            "cache_dir": cache_dir,
            "timestamp_column_name": config.ae.timestamp_column_name
        },
        DFP_DATA_PREP: {
            "module_id": DFP_DATA_PREP,
            "module_name": "dfp_data_prep",
            "namespace": MODULE_NAMESPACE,
            "timestamp_column_name": config.ae.timestamp_column_name,
            "schema": {
                "schema_str": preprocess_schema_str, "encoding": encoding
            }
        },
        DFP_INFERENCE: {
            "module_id": DFP_INFERENCE,
            "module_name": "dfp_inference",
            "namespace": MODULE_NAMESPACE,
            "model_name_formatter": da.model_name_formatter,
            "fallback_username": config.ae.fallback_username,
            "timestamp_column_name": config.ae.timestamp_column_name
        }
    }

    post_proc_config = {
        "module_id": DFP_POST_PROCESSING,
        "module_name": "dfp_post_processing",
        "namespace": MODULE_NAMESPACE,
        "timestamp_column_name": config.ae.timestamp_column_name
    }

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(MultiFileSource(config, filenames=list(kwargs["input_file"])))

    # Here we add a wrapped module that implements the DFP Inference pipeline
    pipeline.add_stage(
        LinearModulesStage(config,
                           module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiMessage))

    pipeline.add_stage(MonitorStage(config, description="Preprocessing & Inference rate", smoothing=0.001))

    pipeline.add_stage(
        FilterDetectionsStage(config, threshold=2.0, filter_source=FilterSource.DATAFRAME, field_name='mean_abs_z'))

    pipeline.add_stage(
        LinearModulesStage(config,
                           post_proc_config,
                           input_port_name="input",
                           output_port_name="output",
                           input_type=MultiMessage,
                           output_type=MultiMessage))

    # Exclude the columns we don't want in our output
    pipeline.add_stage(SerializeStage(config, exclude=['batch_count', 'origin_hash', '_row_hash', '_batch_id']))

    # Write all anomalies to a CSV file
    pipeline.add_stage(WriteToFileStage(config, filename="dfp_detections_azure.csv", overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
