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

import json
import logging
import os
import pickle
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial

import pandas as pd
import pytest
from dfp.messages.multi_dfp_message import MultiDFPMessage
from dfp.stages.multi_file_source import MultiFileSource

from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import create_increment_col
from morpheus.utils.logger import configure_logging

cfd = os.path.dirname(os.path.abspath(__file__))

modules_conf_file = os.path.join(cfd, "modules_config.json")
e2e_test_conf_file = os.path.join(cfd, "e2e_test_configs.json")

with open(modules_conf_file, 'r') as f:
    MODULES_CONFIGS = json.load(f)

with open(e2e_test_conf_file, 'r') as f:
    E2E_TEST_CONFIGS = json.load(f)


def ae_pipeline(config: Config, input_glob):

    configure_logging(log_level=logging.INFO)

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    pipeline.set_source(MultiFileSource(config, filenames=list(input_glob)))

    # Here we add a wrapped module that implements the full DFPPreprocessing pipeline.
    pipeline.add_stage(
        LinearModulesStage(config,
                           MODULES_CONFIGS["preprocessing"],
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiDFPMessage))

    pipeline.add_stage(
        LinearModulesStage(config,
                           MODULES_CONFIGS["training"],
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiDFPMessage))

    pipeline.build()
    pipeline.run()


@pytest.mark.benchmark
def test_dfp_azure_modules_e2e(benchmark, tmp_path):

    config = Config()
    CppConfig.set_should_use_cpp(False)
    config.ae = ConfigAutoEncoder()

    config.num_threads = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["num_threads"]
    config.pipeline_batch_size = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["pipeline_batch_size"]
    config.model_max_batch_size = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["model_max_batch_size"]
    config.feature_length = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["feature_length"]
    config.edge_buffer_size = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["edge_buffer_size"]
    config.ae.userid_column_name = MODULES_CONFIGS["FileToDF"]["userid_column_name"]
    config.ae.timestamp_column_name = MODULES_CONFIGS["FileToDF"]["timestamp_column_name"]
    config.ae.feature_columns = MODULES_CONFIGS["training_module_config"]["DFPTraining"]["feature_columns"]

    input_glob = E2E_TEST_CONFIGS["test_dfp_azure_modules_e2e"]["file_path"]

    start_time = MODULES_CONFIGS["preprocessing"]["FileBatcher"]["start_time"]
    start_time = datetime.strptime(start_time, "%Y-%m-%d")
    duration = MODULES_CONFIGS["preprocessing"]["DFPRollingWindow"]["max_history"]
    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())

    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        end_time = start_time + duration

    MODULES_CONFIGS["preprocessing"]["FileBatcher"]["start_time"] = start_time
    MODULES_CONFIGS["preprocessing"]["FileBatcher"]["end_time"] = end_time

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

    source_schema = DataFrameInputSchema(json_columns=["properties"], column_info=source_column_info)
    preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

    encoding = "latin1"

    # Convert schema as a string
    source_schema_str = str(pickle.dumps(source_schema), encoding=encoding)
    preprocess_schema_str = str(pickle.dumps(preprocess_schema), encoding=encoding)

    MODULES_CONFIGS["preprocessing"]["FileToDF"]["schema"]["schema_str"] = source_schema_str
    MODULES_CONFIGS["preprocessing"]["FileToDF"]["schema"]["encoding"] = encoding
    MODULES_CONFIGS["preprocessing"]["DFPDataPrep"]["schema"]["schema_str"] = preprocess_schema_str
    MODULES_CONFIGS["preprocessing"]["DFPDataPrep"]["schema"]["encoding"] = encoding

    benchmark(ae_pipeline, config, input_glob)
