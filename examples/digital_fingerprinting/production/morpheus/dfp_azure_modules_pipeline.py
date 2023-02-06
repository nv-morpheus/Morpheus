# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial

import click
import dfp.modules.dfp_model_train_deploy  # noqa: F401
import dfp.modules.dfp_preprocessing  # noqa: F401
import mlflow
import pandas as pd
from dfp.messages.multi_dfp_message import MultiDFPMessage
from dfp.stages.dfp_inference_stage import DFPInferenceStage
from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.module_ids import DFP_DATA_PREP
from dfp.utils.module_ids import DFP_MODEL_TRAIN_DEPLOY
from dfp.utils.module_ids import DFP_PREPROCESSING
from dfp.utils.module_ids import DFP_ROLLING_WINDOW
from dfp.utils.module_ids import DFP_SPLIT_USERS
from dfp.utils.module_ids import DFP_TRAINING
from dfp.utils.regex_utils import iso_date_regex_pattern

from morpheus._lib.common import FilterSource
from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
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
from morpheus.utils.logger import configure_logging
from morpheus.utils.module_ids import FILE_BATCHER
from morpheus.utils.module_ids import FILE_TO_DF
from morpheus.utils.module_ids import MLFLOW_MODEL_WRITER
from morpheus.utils.module_ids import MODULE_NAMESPACE


@click.command()
@click.option(
    "--train_users",
    type=click.Choice(["all", "generic", "individual", "none"], case_sensitive=False),
    help=("Indicates whether or not to train per user or a generic model for all users. "
          "Selecting none runs the inference pipeline."),
)
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
    default="60d",
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
def run_pipeline(train_users,
                 skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 start_time: datetime,
                 duration,
                 cache_dir,
                 log_level,
                 sample_rate_s,
                 **kwargs):
    # To include the generic, we must be training all or generic
    include_generic = train_users == "all" or train_users == "generic"

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we arent training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)
    only_users = list(only_user)

    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())
    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        end_time = start_time + duration

    # Enable the Morpheus logger
    configure_logging(log_level=log_level)
    logging.getLogger("mlflow").setLevel(log_level)

    if (len(skip_users) > 0 and len(only_users) > 0):
        logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

    logger = logging.getLogger("morpheus.{}".format(__name__))

    logger.info("Running training pipeline with the following options: ")
    logger.info("Train generic_user: %s", include_generic)
    logger.info("Skipping users: %s", skip_users)
    logger.info("Start Time: %s", start_time)
    logger.info("Duration: %s", duration)
    logger.info("Cache Dir: %s", cache_dir)

    if ("tracking_uri" in kwargs):
        # Initialize ML Flow
        mlflow.set_tracking_uri(kwargs["tracking_uri"])
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    config = Config()

    CppConfig.set_should_use_cpp(False)

    config.num_threads = os.cpu_count()

    config.ae = ConfigAutoEncoder()

    config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_azure.txt"))
    config.ae.userid_column_name = "username"
    config.ae.timestamp_column_name = "timestamp"

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
    source_schema_str = str(pickle.dumps(source_schema), encoding=encoding)
    preprocess_schema_str = str(pickle.dumps(preprocess_schema), encoding=encoding)

    preprocessing_module_config = {
        "module_id": DFP_PREPROCESSING,
        "module_name": "dfp_preprocessing",
        "namespace": MODULE_NAMESPACE,
        FILE_BATCHER: {
            "module_id": FILE_BATCHER,
            "module_name": "file_batcher",
            "namespace": MODULE_NAMESPACE,
            "period": "D",
            "sampling_rate_s": sample_rate_s,
            "start_time": start_time,
            "end_time": end_time,
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
            "include_generic": include_generic,
            "include_individual": include_individual,
            "skip_users": skip_users,
            "only_users": only_users,
            "timestamp_column_name": config.ae.timestamp_column_name,
            "userid_column_name": config.ae.userid_column_name,
            "fallback_username": config.ae.fallback_username
        },
        DFP_ROLLING_WINDOW: {
            "module_id": DFP_ROLLING_WINDOW,
            "module_name": "dfp_rolling_window",
            "namespace": MODULE_NAMESPACE,
            "min_history": 300 if is_training else 1,
            "min_increment": 300 if is_training else 0,
            "max_history": "60d" if is_training else "1d",
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
        }
    }

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(MultiFileSource(config, filenames=list(kwargs["input_file"])))

    # Here we add a wrapped module that implements the full DFPPreprocessing pipeline.
    pipeline.add_stage(
        LinearModulesStage(config,
                           preprocessing_module_config,
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiDFPMessage))

    pipeline.add_stage(MonitorStage(config, description="Preprocessing Module rate", smoothing=0.001))

    model_name_formatter = "DFP-azure-{user_id}"
    experiment_name_formatter = "dfp/azure/training/{reg_model_name}"

    if (is_training):

        # Module configuration
        training_module_config = {
            "module_id": DFP_MODEL_TRAIN_DEPLOY,
            "module_name": "dfp_model_train_deploy",
            "namespace": MODULE_NAMESPACE,
            DFP_TRAINING: {
                "module_id": DFP_TRAINING,
                "module_name": "dfp_training",
                "namespace": MODULE_NAMESPACE,
                "model_kwargs": {
                    "encoder_layers": [512, 500],  # layers of the encoding part
                    "decoder_layers": [512],  # layers of the decoding part
                    "activation": 'relu',  # activation function
                    "swap_p": 0.2,  # noise parameter
                    "lr": 0.001,  # learning rate
                    "lr_decay": 0.99,  # learning decay
                    "batch_size": 512,
                    "verbose": False,
                    "optimizer": 'sgd',  # SGD optimizer is selected(Stochastic gradient descent)
                    "scaler": 'standard',  # feature scaling method
                    "min_cats": 1,  # cut off for minority categories
                    "progress_bar": False,
                    "device": "cuda"
                },
                "feature_columns": config.ae.feature_columns,
                "epochs": 30,
                "validation_size": 0.10
            },
            MLFLOW_MODEL_WRITER: {
                "module_id": MLFLOW_MODEL_WRITER,
                "module_name": "mlflow_model_writer",
                "namespace": MODULE_NAMESPACE,
                "model_name_formatter": model_name_formatter,
                "experiment_name_formatter": experiment_name_formatter,
                "timestamp_column_name": config.ae.timestamp_column_name,
                "conda_env": {
                    'channels': ['defaults', 'conda-forge'],
                    'dependencies': ['python={}'.format('3.8'), 'pip'],
                    'pip': ['mlflow', 'dfencoder'],
                    'name': 'mlflow-env'
                },
                "databricks_permissions": None
            }
        }
        # Here we add a wrapped module that implements the full DFPTraining pipeline.
        pipeline.add_stage(
            LinearModulesStage(
                config,
                training_module_config,
                input_port_name="input",
                output_port_name="output",
            ))

        pipeline.add_stage(MonitorStage(config, description="Training Module rate", smoothing=0.001))

    else:
        # Perform inference on the preprocessed data
        pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=model_name_formatter))

        pipeline.add_stage(MonitorStage(config, description="Inference rate", smoothing=0.001))

        # Filter for only the anomalous logs
        pipeline.add_stage(
            FilterDetectionsStage(config, threshold=2.0, filter_source=FilterSource.DATAFRAME, field_name='mean_abs_z'))

        # Filter for only the anomalous logs
        pipeline.add_stage(DFPPostprocessingStage(config))

        # Exclude the columns we don't want in our output
        pipeline.add_stage(SerializeStage(config, exclude=['batch_count', 'origin_hash', '_row_hash', '_batch_id']))

        # Write all anomalies to a CSV file
        pipeline.add_stage(WriteToFileStage(config, filename="dfp_detections_azure.csv", overwrite=True))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline(obj={}, auto_envvar_prefix='DFP', show_default=True, prog_name="dfp")
