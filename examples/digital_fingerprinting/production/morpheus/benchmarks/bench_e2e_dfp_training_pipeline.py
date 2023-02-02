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

import functools
import glob
import json
import logging
import os
import typing

import dfp.modules.dfp_model_train_deploy  # noqa: F401
import dfp.modules.dfp_preprocessing  # noqa: F401
import pytest
from dfp.messages.multi_dfp_message import MultiDFPMessage
from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage
from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
from dfp.stages.dfp_training import DFPTraining
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.regex_utils import iso_date_regex

from benchmarks.dfp_training_config import DFPTrainingConfig
from benchmarks.dfp_training_config import get_azure_preprocess_schema
from benchmarks.dfp_training_config import get_azure_source_schema
from benchmarks.dfp_training_config import get_duo_preprocess_schema
from benchmarks.dfp_training_config import get_duo_source_schema
from benchmarks.dfp_training_config import set_mlflow_tracking_uri
from morpheus._lib.common import FileTypes
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.logger import configure_logging

curr_file_dir = os.path.dirname(os.path.abspath(__file__))

modules_conf_file = os.path.join(curr_file_dir, "modules_conf.json")
pipelines_conf_file = os.path.join(curr_file_dir, "pipelines_conf.json")

with open(modules_conf_file, 'r') as f:
    MODULES_CONF = json.load(f)

with open(pipelines_conf_file, 'r') as f:
    PIPELINES_CONF = json.load(f)

set_mlflow_tracking_uri(PIPELINES_CONF.get("tracking_uri"))


def dfp_training_pipeline_modules(config: Config, modules_conf: typing.Dict[str, any], filenames: typing.List[str]):

    configure_logging(log_level=logging.INFO)

    pipeline = LinearPipeline(config)
    pipeline.set_source(MultiFileSource(config, filenames=filenames))
    pipeline.add_stage(
        LinearModulesStage(config,
                           modules_conf["preprocessing"],
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiDFPMessage))
    pipeline.add_stage(
        LinearModulesStage(config,
                           modules_conf["train_deploy"],
                           input_port_name="input",
                           output_port_name="output",
                           output_type=MultiDFPMessage))
    pipeline.build()
    pipeline.run()


def dfp_training_pipeline_stages(config: Config,
                                 stages_conf: typing.Dict[str, any],
                                 source_schema: DataFrameInputSchema,
                                 preprocess_schema: DataFrameInputSchema,
                                 filenames: typing.List[str]):

    configure_logging(log_level=logging.INFO)

    pipeline = LinearPipeline(config)
    pipeline.set_source(MultiFileSource(config, filenames=filenames))
    pipeline.add_stage(
        DFPFileBatcherStage(config,
                            period="D",
                            sampling_rate_s=stages_conf["sampling_rate_s"],
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            start_time=stages_conf["start_time"],
                            end_time=stages_conf["end_time"]))
    pipeline.add_stage(
        DFPFileToDataFrameStage(config,
                                schema=source_schema,
                                file_type=FileTypes.JSON,
                                parser_kwargs={
                                    "lines": False, "orient": "records"
                                },
                                cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(
        DFPSplitUsersStage(config,
                           include_generic=stages_conf["include_generic"],
                           include_individual=stages_conf["include_individual"],
                           skip_users=stages_conf["skip_users"],
                           only_users=stages_conf["only_users"]))
    pipeline.add_stage(
        DFPRollingWindowStage(config,
                              min_history=300,
                              min_increment=300,
                              max_history=stages_conf["duration"],
                              cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(DFPPreprocessingStage(config, input_schema=preprocess_schema))
    pipeline.add_stage(DFPTraining(config, validation_size=0.10))
    pipeline.add_stage(
        DFPMLFlowModelWriterStage(config,
                                  model_name_formatter=stages_conf["model_name_formatter"],
                                  experiment_name_formatter=stages_conf["experiment_name_formatter"]))
    pipeline.build()
    pipeline.run()


@pytest.mark.benchmark
@pytest.mark.parametrize("pipeline_name", ["dfp_training_duo_modules_e2e"])
def test_dfp_training_duo_modules_e2e(benchmark: typing.Any, pipeline_name: str):

    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    modules_conf = MODULES_CONF.copy()
    pipeline_conf = PIPELINES_CONF.get(pipeline_name)

    dfp_tc = DFPTrainingConfig(pipeline_conf, feature_columns, source="duo", modules_conf=modules_conf)
    config: Config = dfp_tc.get_config()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    dfp_tc.update_modules_conf(source_schema, preprocess_schema)

    filenames = glob.glob(pipeline_conf.get("file_path"))

    benchmark(dfp_training_pipeline_modules, config, dfp_tc.modules_conf, filenames)


@pytest.mark.benchmark
@pytest.mark.parametrize("pipeline_name", ["dfp_training_duo_stages_e2e"])
def test_dfp_training_duo_stages_e2e(benchmark: typing.Any, pipeline_name: str):

    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    pipeline_conf = PIPELINES_CONF.get(pipeline_name)

    dfp_tc = DFPTrainingConfig(pipeline_conf, feature_columns, source="duo")
    config: Config = dfp_tc.get_config()
    stages_conf = dfp_tc.get_stages_conf()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    filenames = glob.glob(pipeline_conf.get("file_path"))

    benchmark(dfp_training_pipeline_stages, config, stages_conf, source_schema, preprocess_schema, filenames)


@pytest.mark.benchmark
@pytest.mark.parametrize("pipeline_name", ["dfp_training_azure_modules_e2e"])
def test_dfp_training_azure_modules_e2e(benchmark: typing.Any, pipeline_name: str):

    feature_columns = [
        "appDisplayName",
        "clientAppUsed",
        "deviceDetailbrowser",
        "deviceDetaildisplayName",
        "deviceDetailoperatingSystem",
        "statusfailureReason",
        "appincrement",
        "locincrement",
        "logcount"
    ]

    modules_conf = MODULES_CONF.copy()
    pipeline_conf = PIPELINES_CONF.get(pipeline_name)

    dfp_tc = DFPTrainingConfig(pipeline_conf, feature_columns, source="azure", modules_conf=modules_conf)
    config: Config = dfp_tc.get_config()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    dfp_tc.update_modules_conf(source_schema, preprocess_schema)

    filenames = glob.glob(pipeline_conf.get("file_path"))

    benchmark(dfp_training_pipeline_modules, config, dfp_tc.modules_conf, filenames)


@pytest.mark.benchmark
@pytest.mark.parametrize("pipeline_name", ["dfp_training_azure_stages_e2e"])
def test_dfp_training_azure_stages_e2e(benchmark: typing.Any, pipeline_name: str):

    feature_columns = [
        "appDisplayName",
        "clientAppUsed",
        "deviceDetailbrowser",
        "deviceDetaildisplayName",
        "deviceDetailoperatingSystem",
        "statusfailureReason",
        "appincrement",
        "locincrement",
        "logcount"
    ]

    pipeline_conf = PIPELINES_CONF.get(pipeline_name)

    dfp_tc = DFPTrainingConfig(pipeline_conf, feature_columns, source="azure")
    config: Config = dfp_tc.get_config()
    stages_conf = dfp_tc.get_stages_conf()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    filenames = glob.glob(pipeline_conf.get("file_path"))

    benchmark(dfp_training_pipeline_stages, config, stages_conf, source_schema, preprocess_schema, filenames)
