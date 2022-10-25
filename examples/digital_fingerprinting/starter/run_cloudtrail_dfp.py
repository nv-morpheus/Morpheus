# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from datetime import datetime
import functools
import logging
import os
import typing

import click
import mlflow
from morpheus.stages.inference.dfp_inference_stage import DFPInferenceStage
from morpheus.stages.preprocess.dfp_training import DFPTraining
from morpheus.stages.input.dfp_split_users_stage import DFPSplitUsersStage
from morpheus.stages.preprocess.dfp_rolling_window_stage import DFPRollingWindowStage

from morpheus.config import AEFeatureScalar
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus._lib.file_types import FileTypes
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.multi_file_source_stage import MultiFileSourceStage
from morpheus.stages.input.dfp_file_batcher_stage import DFPFileBatcherStage
from morpheus.stages.output.write_dfp_mlflow_model import WriteDFPMLFlowStage
from morpheus.stages.input.dfp_file_to_df import DFPFileToDataFrameStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.column_info import ColumnInfo, CustomColumn, DataFrameInputSchema, DateTimeColumn, RenameColumn, cast_to_str, clean_column
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.file_utils import iso_date_regex
from morpheus.utils.logger import configure_logging

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
    default=1024,
    type=click.IntRange(min=1),
    help="Max batch size to use for the model",
)
@click.option(
    "--columns_file",
    type=click.Path(exists=True, readable=True),
    required=True,
    help="Feature columns file",
)
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
@click.option(
    "--file_format",
    "-ft",
    type=str,
    multiple=False,
    help=("File types such as `json` or `csv`"),
)
@click.option(
    "--train_data_file",
    "-f",
    type=str,
    multiple=True,
    help=("List of files to process. Can specify multiple arguments for multiple files. "
          "Also accepts glob (*) wildcards and schema prefixes such as `s3://`. "
          "For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. "
          "See fsspec documentation for list of possible options."),
)
@click.option(
    "--pretrained_filename",
    type=click.Path(exists=True, readable=True),
    required=False,
    help="File with pre-trained user models",
)
@click.option(
    "--models_output_filename",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--output_file",
    default="./cloudtrail-detections.csv",
    help="The path to the file where the inference output will be saved.",
)
@click.option(
    "--cache_dir",
    type=str,
    default="./.cache/dfp",
    show_envvar=True,
    help="The location to cache data such as S3 downloads and pre-processed data",
)
@click.option('--tracking_uri',
              type=str,
              default="http://localhost:5000",
              help=("The MLflow tracking URI to connect to the tracking backend."))
def run_pipeline(train_users,
                 num_threads,
                 pipeline_batch_size,
                 model_max_batch_size,
                 columns_file,
                 output_file,
                 cache_dir,
                 file_format,
                 skip_user: typing.Tuple[str],
                 only_user: typing.Tuple[str],
                 **kwargs):
    
    # To include the generic, we must be training all or generic
    include_generic = train_users == "all" or train_users == "generic"

    # To include individual, we must be either training or inferring
    include_individual = train_users != "generic"

    # None indicates we arent training anything
    is_training = train_users != "none"

    skip_users = list(skip_user)
    only_users = list(only_user)

    configure_logging(log_level=logging.DEBUG)

    if (len(skip_users) > 0 and len(only_users) > 0):
        logging.error("Option --skip_user and --only_user are mutually exclusive. Exiting")

    logger = logging.getLogger("morpheus.{}".format(__name__))

    logger.info("Running training pipeline with the following options: ")
    logger.info("Train generic_user: %s", include_generic)
    logger.info("Skipping users: %s", skip_users)
    logger.info("Cache Dir: %s", cache_dir)

    if ("tracking_uri" in kwargs):
        # Initialize ML Flow
        mlflow.set_tracking_uri(kwargs["tracking_uri"])
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.mode = PipelineModes.AE
    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.feature_scaler = AEFeatureScalar.STANDARD

    with open(columns_file, "r") as lf:
        config.ae.feature_columns = [x.strip() for x in lf.readlines()]

    config.num_threads = num_threads
    config.pipeline_batch_size = pipeline_batch_size
    config.model_max_batch_size = model_max_batch_size

    if file_format.upper() == "JSON":
        file_type = FileTypes.JSON
        parser_kwargs={"lines": False, "orient": "records"}

        source_column_info = [
            DateTimeColumn(name="timestamp", dtype=datetime, input_name="eventTime"),
            ColumnInfo(name="eventSource", dtype=str),
            ColumnInfo(name="eventName", dtype=str),
            ColumnInfo(name="sourceIPAddress", dtype=bool),
            ColumnInfo(name="userAgent", dtype=str),
            RenameColumn(name="userIdentitytype", dtype=str, input_name="userIdentity.type"),
            RenameColumn(name="requestParametersroleArn", dtype=str, input_name="requestParameters.roleArn"),
            RenameColumn(name="requestParametersroleSessionName", dtype=str, input_name="requestParameters.roleSessionName"),
            RenameColumn(name="requestParametersdurationSeconds", dtype=str, input_name="requestParameters.durationSeconds"),
            RenameColumn(name="responseElementsassumedRoleUserassumedRoleId", dtype=str, input_name="responseElements.assumedRoleUser.assumedRoleId"),
            RenameColumn(name="responseElementsassumedRoleUserarn", dtype=str, input_name="responseElements.assumedRoleUser.arn"),
            ColumnInfo(name="apiVersion", dtype=str),
            CustomColumn(name="userIdentityprincipalId",
                        dtype=str,
                        process_column_fn=functools.partial(cast_to_str, column_name="userIdentity.principalId", prefix="Account-")),
            RenameColumn(name="userIdentityarn", dtype=str, input_name="userIdentity.arn"),
            RenameColumn(name="userIdentityaccountId", dtype=str, input_name="userIdentity.accountId"),
            RenameColumn(name="userIdentityaccessKeyId", dtype=str, input_name="userIdentity.accessKeyId"),
            RenameColumn(name="userIdentitysessionContextsessionIssuerprincipalId", dtype=str, input_name="userIdentity.sessionContextsession.IssuerprincipalId"),
            ColumnInfo(name=config.ae.userid_column_name, dtype=str),
            ColumnInfo(name="tlsDetailsclientProvidedHostHeader", dtype=str),
            RenameColumn(name="requestParametersownersSetitems", dtype=str, input_name="requestParameters.ownersSetitems"),
            RenameColumn(name="requestParametersmaxResults", dtype=str, input_name="requestParameters.maxResults"),
            RenameColumn(name="requestParametersinstancesSetitems", dtype=str, input_name="requestParameters.instancesSetitems"),
            ColumnInfo(name="errorCode", dtype=str),
            ColumnInfo(name="errorMessage", dtype=str),
            RenameColumn(name="requestParametersmaxItems", dtype=str, input_name="requestParameters.maxItems"),
            RenameColumn(name="responseElementsrequestId", dtype=str, input_name="responseElements.requestId"),
            RenameColumn(name="responseElementsinstancesSetitems", dtype=str, input_name="responseElements.instancesSetitems"),
            RenameColumn(name="requestParametersgroupSetitems", dtype=str, input_name="requestParameters.groupSetitems"),
            RenameColumn(name="requestParametersinstanceType", dtype=str, input_name="requestParameters.instanceType"),
            RenameColumn(name="requestParametersmonitoringenabled", dtype=str, input_name="requestParameters.monitoringenabled"),
            RenameColumn(name="requestParametersdisableApiTermination", dtype=str, input_name="requestParameters.disableApiTermination"),
            RenameColumn(name="requestParametersebsOptimized", dtype=str, input_name="requestParameters.ebsOptimized"),
            RenameColumn(name="responseElementsreservationId", dtype=str, input_name="responseElements.reservationId"),
            RenameColumn(name="requestParametersgroupName", dtype=str, input_name="requestParameters.groupName"),
            ]

    elif file_format.upper() == "CSV":
        file_type = FileTypes.CSV
        parser_kwargs = {}

        source_column_info = [
            DateTimeColumn(name="timestamp", dtype=datetime, input_name="eventTime"),
            ColumnInfo(name="eventSource", dtype=str),
            ColumnInfo(name="eventName", dtype=str),
            ColumnInfo(name="sourceIPAddress", dtype=bool),
            ColumnInfo(name="userAgent", dtype=str),
            ColumnInfo(name="userIdentitytype", dtype=str),
            ColumnInfo(name="requestParametersroleArn", dtype=str),
            ColumnInfo(name="requestParametersroleSessionName", dtype=str),
            ColumnInfo(name="requestParametersdurationSeconds", dtype=str),
            ColumnInfo(name="assumedRoleUserassumedRoleId", dtype=str),
            ColumnInfo(name="responseElementsassumedRoleUserarn", dtype=str),
            ColumnInfo(name="apiVersion", dtype=str),
            CustomColumn(name="userIdentityprincipalId",
                        dtype=str,
                        process_column_fn=functools.partial(cast_to_str, column_name="userIdentityprincipalId", prefix="Account-")),
            ColumnInfo(name="userIdentityarn", dtype=str),
            ColumnInfo(name="userIdentityaccountId", dtype=str),
            ColumnInfo(name="userIdentityaccessKeyId", dtype=str),
            ColumnInfo(name="userIdentitysessionContextsessionIssuerprincipalId", dtype=str),
            ColumnInfo(name=config.ae.userid_column_name, dtype=str),
            ColumnInfo(name="tlsDetailsclientProvidedHostHeader", dtype=str),
            ColumnInfo(name="requestParametersownersSetitems", dtype=str),
            ColumnInfo(name="requestParametersmaxResults", dtype=str),
            ColumnInfo(name="requestParametersinstancesSetitems", dtype=str),
            ColumnInfo(name="errorCode", dtype=str),
            ColumnInfo(name="errorMessage", dtype=str),
            ColumnInfo(name="requestParametersmaxItems", dtype=str),
            ColumnInfo(name="responseElementsrequestId", dtype=str),
            ColumnInfo(name="responseElementsinstancesSetitems", dtype=str),
            ColumnInfo(name="requestParametersgroupSetitems", dtype=str),
            ColumnInfo(name="requestParametersinstanceType", dtype=str),
            ColumnInfo(name="requestParametersmonitoringenabled", dtype=str),
            ColumnInfo(name="requestParametersdisableApiTermination", dtype=str),
            ColumnInfo(name="requestParametersebsOptimized", dtype=str),
            ColumnInfo(name="responseElementsreservationId", dtype=str),
            ColumnInfo(name="requestParametersgroupName", dtype=str),
            ]
    else:
        "Unsupported file format: {}".format(file_format)
    
    # Specify the column names to ensure all data is uniform
    
    
    source_schema = DataFrameInputSchema(column_info=source_column_info)

    # Create a pipeline object
    pipeline = LinearPipeline(config)

    # Add a source stage
    pipeline.set_source(MultiFileSourceStage(config, filenames=list(kwargs["input_file"])))

    # Batch files into buckets by time. Use the default ISO date extractor from the filename
    pipeline.add_stage(
        DFPFileBatcherStage(config,
                            period="D",
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            no_time_window=True))

    # Output is S3 Buckets. Convert to DataFrames. This caches downloaded S3 data
    pipeline.add_stage(
        DFPFileToDataFrameStage(config,
                                schema=source_schema,
                                file_type=file_type,
                                parser_kwargs=parser_kwargs,
                                cache_dir=cache_dir))
    
    # This will split users or just use one single user
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=include_generic,
                           include_individual=include_individual,
                           skip_users=skip_users,
                           only_users=only_users))
    
    # Next, have a stage that will create rolling windows
    pipeline.add_stage(
        DFPRollingWindowStage(
            config,
            min_history=300 if is_training else 1,
            min_increment=300 if is_training else 0,
            # For inference, we only ever want 1 day max
            max_history="60d" if is_training else "1d",
            cache_dir=cache_dir))
    


    model_name_formatter = "DFP-cloudtrail-{user_id}"
    experiment_name_formatter = "dfp/cloudtrail/training/{reg_model_name}"

    if (is_training):

        # Finally, perform training which will output a model
        pipeline.add_stage(DFPTraining(config))

        pipeline.add_stage(MonitorStage(config, description="Training rate", smoothing=0.001))

        # Write that model to MLFlow
        pipeline.add_stage(
            WriteDFPMLFlowStage(config,
                                      model_name_formatter=model_name_formatter,
                                      experiment_name_formatter=experiment_name_formatter))
    else:
        pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=model_name_formatter))

        pipeline.add_stage(MonitorStage(config, description="Inference rate", smoothing=0.001))

        # Add serialize stage
        pipeline.add_stage(SerializeStage(config))

        # Add a write file stage
        pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

        pipeline.add_stage(MonitorStage(config, description="WriteToFile rate"))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
