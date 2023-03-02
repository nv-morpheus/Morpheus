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
import logging
import os
import typing

# flake8 warnings are silenced by the addition of noqa.
import dfp.modules.dfp_deployment  # noqa: F401
import pytest
from dfp.stages.dfp_file_batcher_stage import DFPFileBatcherStage
from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
from dfp.stages.dfp_inference_stage import DFPInferenceStage
from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
from dfp.stages.dfp_postprocessing_stage import DFPPostprocessingStage
from dfp.stages.dfp_preprocessing_stage import DFPPreprocessingStage
from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
from dfp.stages.dfp_split_users_stage import DFPSplitUsersStage
from dfp.stages.dfp_training import DFPTraining
from dfp.stages.multi_file_source import MultiFileSource
from dfp.utils.regex_utils import iso_date_regex

from benchmarks.dfp_config import DFPConfig
from benchmarks.dfp_config import get_azure_preprocess_schema
from benchmarks.dfp_config import get_azure_source_schema
from benchmarks.dfp_config import get_duo_preprocess_schema
from benchmarks.dfp_config import get_duo_source_schema
from benchmarks.dfp_config import load_json
from benchmarks.dfp_config import set_mlflow_tracking_uri
from morpheus._lib.common import FileTypes
from morpheus._lib.common import FilterSource
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.pipeline import Pipeline  # noqa: F401
from morpheus.stages.general.multi_port_module_stage import MultiPortModuleStage
from morpheus.stages.input.control_message_source_stage import ControlMessageSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.logger import configure_logging

MODULES_CONF = load_json("resource/modules_conf.json")
PIPELINES_CONF = load_json("resource/pipelines_conf.json")

set_mlflow_tracking_uri(PIPELINES_CONF.get("tracking_uri"))


def dfp_modules_pipeline(config: Config, modules_conf: typing.Dict[str, any], filenames: typing.List[str]):
    configure_logging(log_level=logging.CRITICAL)

    pipeline = Pipeline(config)

    source_stage = pipeline.add_stage(ControlMessageSourceStage(config, filenames=filenames))

    import json
    with open("modules_conf.json", "w") as f:
        f.write(json.dumps(modules_conf, indent=3, default=str))

    # Here we add a wrapped module that implements the DFP Deployment
    dfp_deployment_stage = pipeline.add_stage(
        MultiPortModuleStage(config,
                             modules_conf,
                             input_port_name="input",
                             output_port_name_prefix="output",
                             output_port_count=modules_conf.get("output_port_count")))

    pipeline.add_edge(source_stage, dfp_deployment_stage)

    pipeline.run()


def dfp_training_pipeline_stages(config: Config,
                                 stages_conf: typing.Dict[str, any],
                                 source_schema: DataFrameInputSchema,
                                 preprocess_schema: DataFrameInputSchema,
                                 filenames: typing.List[str]):
    configure_logging(log_level=logging.CRITICAL)

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


def dfp_inference_pipeline_stages(config: Config,
                                  stages_conf: typing.Dict[str, any],
                                  source_schema: DataFrameInputSchema,
                                  preprocess_schema: DataFrameInputSchema,
                                  filenames: typing.List[str],
                                  output_filepath: str):
    configure_logging(log_level=logging.CRITICAL)

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
                              min_history=1,
                              min_increment=0,
                              max_history=stages_conf["duration"],
                              cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(DFPPreprocessingStage(config, input_schema=preprocess_schema))
    pipeline.add_stage(DFPInferenceStage(config, model_name_formatter=stages_conf["model_name_formatter"]))
    pipeline.add_stage(
        FilterDetectionsStage(config, threshold=2.0, filter_source=FilterSource.DATAFRAME, field_name='mean_abs_z'))
    pipeline.add_stage(DFPPostprocessingStage(config))
    pipeline.add_stage(SerializeStage(config, exclude=['batch_count', 'origin_hash', '_row_hash', '_batch_id']))
    pipeline.add_stage(WriteToFileStage(config, filename=output_filepath, overwrite=True))

    pipeline.build()
    pipeline.run()


@pytest.mark.benchmark
def test_dfp_training_duo_stages_e2e(benchmark: typing.Any):
    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    pipeline_conf = PIPELINES_CONF.get("test_dfp_training_duo_stages_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="duo")

    config = dfp_config.get_config()
    stages_conf = dfp_config.get_stages_conf()
    filenames = dfp_config.get_filenames()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    benchmark(dfp_training_pipeline_stages, config, stages_conf, source_schema, preprocess_schema, filenames)


@pytest.mark.benchmark
def test_dfp_training_azure_stages_e2e(benchmark: typing.Any):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_training_azure_stages_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="azure")

    config = dfp_config.get_config()
    stages_conf = dfp_config.get_stages_conf()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    benchmark(dfp_training_pipeline_stages, config, stages_conf, source_schema, preprocess_schema, filenames)


@pytest.mark.benchmark
def test_dfp_inference_azure_stages_e2e(benchmark: typing.Any, tmp_path):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_inference_azure_stages_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="azure")

    config = dfp_config.get_config()
    stages_conf = dfp_config.get_stages_conf()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    output_filepath = os.path.join(tmp_path, "detections_azure.csv")

    benchmark(dfp_inference_pipeline_stages,
              config,
              stages_conf,
              source_schema,
              preprocess_schema,
              filenames,
              output_filepath)


@pytest.mark.benchmark
def test_dfp_inference_duo_stages_e2e(benchmark: typing.Any, tmp_path):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_inference_duo_stages_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="duo")

    config = dfp_config.get_config()
    stages_conf = dfp_config.get_stages_conf()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    output_filepath = os.path.join(tmp_path, "detections_duo.csv")

    benchmark(dfp_inference_pipeline_stages,
              config,
              stages_conf,
              source_schema,
              preprocess_schema,
              filenames,
              output_filepath)


@pytest.mark.benchmark
def test_dfp_modules_duo_training_e2e(benchmark: typing.Any):
    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_duo_training_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="duo", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_training_e2e(benchmark: typing.Any):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_azure_training_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="azure", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_inference_e2e(benchmark: typing.Any):
    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_duo_inference_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="duo", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_inference_e2e(benchmark: typing.Any):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_azure_inference_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="azure", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_e2e(benchmark: typing.Any):
    feature_columns = [
        "accessdevicebrowser",
        "accessdeviceos",
        "authdevicename",
        "reason",
        "result",
        "locincrement",
        "logcount",
    ]

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_duo_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="duo", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_duo_source_schema(config)
    preprocess_schema = get_duo_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_e2e(benchmark: typing.Any):
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

    pipeline_conf = PIPELINES_CONF.get("test_dfp_modules_azure_e2e")

    dfp_config = DFPConfig(pipeline_conf, feature_columns, source="azure", modules_conf=MODULES_CONF)

    config = dfp_config.get_config()
    filenames = dfp_config.get_filenames()

    source_schema = get_azure_source_schema(config)
    preprocess_schema = get_azure_preprocess_schema(config)

    dfp_config.update_modules_conf(source_schema, preprocess_schema)

    benchmark(dfp_modules_pipeline, config, dfp_config.modules_conf, filenames)
