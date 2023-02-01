# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pickle
import typing
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial

import mlflow
import pandas as pd

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import create_increment_col


def get_updated_pipeline_config(pipeline_conf: typing.Dict[str, any], feature_columns: typing.List[str]) -> Config:

    config = Config()
    CppConfig.set_should_use_cpp(False)
    config.ae = ConfigAutoEncoder()

    config.num_threads = pipeline_conf["num_threads"]
    config.pipeline_batch_size = pipeline_conf["pipeline_batch_size"]
    config.edge_buffer_size = pipeline_conf["edge_buffer_size"]
    config.ae.userid_column_name = "username"
    config.ae.timestamp_column_name = "timestamp"
    config.ae.feature_columns = feature_columns

    return config


def get_duo_source_schema(config: Config) -> DataFrameInputSchema:

    # Source schema
    source_column_info = [
        DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name="timestamp"),
        RenameColumn(name=config.ae.userid_column_name, dtype=str, input_name="user.name"),
        RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
        RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "access_device.location.city",
                            "access_device.location.state",
                            "access_device.location.country"
                        ],
                        sep=", "),
        RenameColumn(name="authdevicename", dtype=str, input_name="auth_device.name"),
        BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["success", "SUCCESS"],
                   false_values=["denied", "DENIED", "FRAUD"]),
        ColumnInfo(name="reason", dtype=str),
    ]

    source_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                         column_info=source_column_info)

    return source_schema


def get_duo_preprocess_schema(config: Config) -> DataFrameInputSchema:

    # Preprocessing schema
    preprocess_column_info = [
        ColumnInfo(name=config.ae.timestamp_column_name, dtype=datetime),
        ColumnInfo(name=config.ae.userid_column_name, dtype=str),
        ColumnInfo(name="accessdevicebrowser", dtype=str),
        ColumnInfo(name="accessdeviceos", dtype=str),
        ColumnInfo(name="authdevicename", dtype=str),
        ColumnInfo(name="result", dtype=bool),
        ColumnInfo(name="reason", dtype=str),
        IncrementColumn(name="logcount",
                        dtype=int,
                        input_name=config.ae.timestamp_column_name,
                        groupby_column=config.ae.userid_column_name),
        CustomColumn(name="locincrement",
                     dtype=int,
                     process_column_fn=partial(create_increment_col, column_name="location")),
    ]

    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    return preprocess_schema


def get_azure_source_schema(config: Config) -> DataFrameInputSchema:

    # Source schema
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

    return source_schema


def get_azure_preprocess_schema(config: Config) -> DataFrameInputSchema:

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

    return preprocess_schema


def get_start_stop_time(pipeline_conf: typing.Dict[str, any]) -> typing.Tuple[datetime, datetime]:
    start_time = pipeline_conf["start_time"]
    start_time = datetime.strptime(start_time, "%Y-%m-%d")

    duration = pipeline_conf["duration"]

    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())

    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        end_time = start_time + duration
    return tuple((start_time, end_time))


def set_mlflow_tracking_uri(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    logging.getLogger("mlflow").setLevel(logging.WARN)


def get_model_name_formatter(source: str) -> str:
    model_name_formatter = "DFP-{}-".format(source) + "{user_id}"
    return model_name_formatter


def get_experiment_name_formatter(source) -> str:
    experiment_name_formatter = "dfp/{}/training/".format(source) + "{reg_model_name}"
    return experiment_name_formatter


def get_updated_modules_conf(pipeline_conf: typing.Dict[str, any],
                             modules_conf: typing.Dict[str, any],
                             source_schema: DataFrameInputSchema,
                             preprocess_schema: DataFrameInputSchema,
                             feature_columns: typing.List[str],
                             source: str) -> typing.Dict[str, any]:

    start_stop_time = get_start_stop_time(pipeline_conf=pipeline_conf)
    modules_conf["preprocessing"]["FileBatcher"]["start_time"] = start_stop_time[0]
    modules_conf["preprocessing"]["FileBatcher"]["end_time"] = start_stop_time[0]
    modules_conf["preprocessing"]["DFPRollingWindow"]["max_history"] = pipeline_conf["duration"]

    encoding = "latin1"

    # Convert schema as a string
    source_schema_str = str(pickle.dumps(source_schema), encoding=encoding)
    preprocess_schema_str = str(pickle.dumps(preprocess_schema), encoding=encoding)

    modules_conf["preprocessing"]["FileToDF"]["schema"]["schema_str"] = source_schema_str
    modules_conf["preprocessing"]["FileToDF"]["schema"]["encoding"] = encoding
    modules_conf["preprocessing"]["DFPDataPrep"]["schema"]["schema_str"] = preprocess_schema_str
    modules_conf["preprocessing"]["DFPDataPrep"]["schema"]["encoding"] = encoding
    modules_conf["train_deploy"]["DFPTraining"]["feature_columns"] = feature_columns

    modules_conf["train_deploy"]["MLFlowModelWriter"]["model_name_formatter"] = get_model_name_formatter(source)
    modules_conf["train_deploy"]["MLFlowModelWriter"]["experiment_name_formatter"] = get_experiment_name_formatter(
        source)

    return modules_conf


def get_stages_conf(pipeline_conf: typing.Dict[str, any], source: str) -> typing.Dict[str, any]:
    stages_conf = {}
    start_stop_time = get_start_stop_time(pipeline_conf=pipeline_conf)
    stages_conf["start_time"] = start_stop_time[0]
    stages_conf["end_time"] = start_stop_time[1]
    stages_conf["duration"] = pipeline_conf["duration"]
    stages_conf["sampling_rate_s"] = 0
    stages_conf["cache_dir"] = "./.cache/dfp"
    stages_conf["include_generic"] = True
    stages_conf["include_individual"] = []
    stages_conf["skip_users"] = []
    stages_conf["only_users"] = []
    stages_conf["model_name_formatter"] = get_model_name_formatter(source)
    stages_conf["experiment_name_formatter"] = get_experiment_name_formatter(source)

    return stages_conf
