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
import shutil
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
from dfp.utils.schema_utils import Schema

from benchmarks.benchmark_conf_generator import BenchmarkConfGenerator
from benchmarks.benchmark_conf_generator import load_json
from benchmarks.benchmark_conf_generator import set_mlflow_tracking_uri
from morpheus._lib.common import FileTypes
from morpheus._lib.common import FilterSource
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.multi_port_module_stage import MultiPortModuleStage
from morpheus.stages.input.control_message_source_stage import ControlMessageSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.logger import configure_logging

logger = logging.getLogger("morpheus.{}".format(__name__))

PIPELINES_CONF = load_json("resource/pipelines_conf.json")

TRACKING_URI = PIPELINES_CONF.get("tracking_uri")

set_mlflow_tracking_uri(PIPELINES_CONF.get("tracking_uri"))


def remove_cache(dir: str):
    logger.debug(f"Cleaning up cache `{dir}` directory...")
    shutil.rmtree(dir, ignore_errors=True)
    logger.debug(f"Cleaning up cache `{dir}` directory... Done")


def dfp_modules_pipeline(pipe_config: Config,
                         modules_conf: typing.Dict[str, any],
                         filenames: typing.List[str],
                         reuse_cache=False):

    pipeline = Pipeline(pipe_config)

    source_stage = pipeline.add_stage(ControlMessageSourceStage(pipe_config, filenames=filenames))

    # Here we add a wrapped module that implements the DFP Deployment
    dfp_deployment_stage = pipeline.add_stage(
        MultiPortModuleStage(pipe_config,
                             modules_conf,
                             input_port_name="input",
                             output_port_name_prefix="output",
                             output_port_count=modules_conf["output_port_count"]))

    pipeline.add_edge(source_stage, dfp_deployment_stage)

    pipeline.run()

    if not reuse_cache:
        cache_dir = modules_conf["DFPInferencePipe"]["DFPPreproc"]["FileBatcher"]["cache_dir"]
        remove_cache(dir=cache_dir)


def dfp_training_pipeline_stages(pipe_config: Config,
                                 stages_conf: typing.Dict[str, any],
                                 source_schema: DataFrameInputSchema,
                                 preprocess_schema: DataFrameInputSchema,
                                 filenames: typing.List[str],
                                 log_level: int,
                                 reuse_cache=False):

    configure_logging(log_level)

    pipeline = LinearPipeline(pipe_config)
    pipeline.set_source(MultiFileSource(pipe_config, filenames=filenames))
    pipeline.add_stage(
        DFPFileBatcherStage(pipe_config,
                            period="D",
                            sampling_rate_s=stages_conf["sampling_rate_s"],
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            start_time=stages_conf["start_time"],
                            end_time=stages_conf["end_time"]))
    pipeline.add_stage(
        DFPFileToDataFrameStage(pipe_config,
                                schema=source_schema,
                                file_type=FileTypes.JSON,
                                parser_kwargs={
                                    "lines": False, "orient": "records"
                                },
                                cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(
        DFPSplitUsersStage(pipe_config,
                           include_generic=stages_conf["include_generic"],
                           include_individual=stages_conf["include_individual"],
                           skip_users=stages_conf["skip_users"],
                           only_users=stages_conf["only_users"]))
    pipeline.add_stage(
        DFPRollingWindowStage(pipe_config,
                              min_history=300,
                              min_increment=300,
                              max_history=stages_conf["duration"],
                              cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(DFPPreprocessingStage(pipe_config, input_schema=preprocess_schema))
    pipeline.add_stage(DFPTraining(pipe_config, validation_size=0.10))
    pipeline.add_stage(
        DFPMLFlowModelWriterStage(pipe_config,
                                  model_name_formatter=stages_conf["model_name_formatter"],
                                  experiment_name_formatter=stages_conf["experiment_name_formatter"]))
    pipeline.build()
    pipeline.run()

    if not reuse_cache:
        remove_cache(dir=stages_conf["cache_dir"])


def dfp_inference_pipeline_stages(pipe_config: Config,
                                  stages_conf: typing.Dict[str, any],
                                  source_schema: DataFrameInputSchema,
                                  preprocess_schema: DataFrameInputSchema,
                                  filenames: typing.List[str],
                                  output_filepath: str,
                                  log_level: int,
                                  reuse_cache=False):

    configure_logging(log_level)

    pipeline = LinearPipeline(pipe_config)
    pipeline.set_source(MultiFileSource(pipe_config, filenames=filenames))
    pipeline.add_stage(
        DFPFileBatcherStage(pipe_config,
                            period="D",
                            sampling_rate_s=stages_conf["sampling_rate_s"],
                            date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex),
                            start_time=stages_conf["start_time"],
                            end_time=stages_conf["end_time"]))
    pipeline.add_stage(
        DFPFileToDataFrameStage(pipe_config,
                                schema=source_schema,
                                file_type=FileTypes.JSON,
                                parser_kwargs={
                                    "lines": False, "orient": "records"
                                },
                                cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(
        DFPSplitUsersStage(pipe_config,
                           include_generic=stages_conf["include_generic"],
                           include_individual=stages_conf["include_individual"],
                           skip_users=stages_conf["skip_users"],
                           only_users=stages_conf["only_users"]))
    pipeline.add_stage(
        DFPRollingWindowStage(pipe_config,
                              min_history=1,
                              min_increment=0,
                              max_history=stages_conf["duration"],
                              cache_dir=stages_conf["cache_dir"]))
    pipeline.add_stage(DFPPreprocessingStage(pipe_config, input_schema=preprocess_schema))
    pipeline.add_stage(DFPInferenceStage(pipe_config, model_name_formatter=stages_conf["model_name_formatter"]))
    pipeline.add_stage(
        FilterDetectionsStage(pipe_config, threshold=2.0, filter_source=FilterSource.DATAFRAME,
                              field_name='mean_abs_z'))
    pipeline.add_stage(DFPPostprocessingStage(pipe_config))
    pipeline.add_stage(SerializeStage(pipe_config, exclude=['batch_count', 'origin_hash', '_row_hash', '_batch_id']))
    pipeline.add_stage(WriteToFileStage(pipe_config, filename=output_filepath, overwrite=True))

    pipeline.build()
    pipeline.run()

    if not reuse_cache:
        remove_cache(dir=stages_conf["cache_dir"])


@pytest.mark.benchmark
def test_dfp_stages_duo_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_stages_duo_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    stages_conf = bcg.get_stages_conf()
    input_filenames = bcg.get_filenames()
    schema: Schema = bcg.get_schema()

    benchmark(dfp_training_pipeline_stages,
              pipe_config,
              stages_conf,
              schema.source,
              schema.preprocess,
              input_filenames,
              bcg.log_level)


@pytest.mark.benchmark
def test_dfp_stages_azure_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_stages_azure_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    stages_conf = bcg.get_stages_conf()
    input_filenames = bcg.get_filenames()
    schema: Schema = bcg.get_schema()

    benchmark(dfp_training_pipeline_stages,
              pipe_config,
              stages_conf,
              schema.source,
              schema.preprocess,
              input_filenames,
              bcg.log_level)


@pytest.mark.benchmark
def test_dfp_stages_azure_inference_e2e(benchmark: typing.Any, tmp_path):

    pipe_conf = PIPELINES_CONF.get("test_dfp_stages_azure_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    stages_conf = bcg.get_stages_conf()
    input_filenames = bcg.get_filenames()
    schema: Schema = bcg.get_schema()

    output_filepath = os.path.join(tmp_path, "detections_azure.csv")

    benchmark(dfp_inference_pipeline_stages,
              pipe_config,
              stages_conf,
              schema.source,
              schema.preprocess,
              input_filenames,
              output_filepath,
              bcg.log_level)


@pytest.mark.benchmark
def test_dfp_stages_duo_inference_e2e(benchmark: typing.Any, tmp_path):

    pipe_conf = PIPELINES_CONF.get("test_dfp_stages_duo_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    stages_conf = bcg.get_stages_conf()
    input_filenames = bcg.get_filenames()
    schema: Schema = bcg.get_schema()

    output_filepath = os.path.join(tmp_path, "detections_duo.csv")

    benchmark(dfp_inference_pipeline_stages,
              pipe_config,
              stages_conf,
              schema.source,
              schema.preprocess,
              input_filenames,
              output_filepath,
              bcg.log_level)


@pytest.mark.benchmark
def test_dfp_modules_azure_payload_inference_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_payload_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_payload_lti_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_payload_lti_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_payload_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_payload_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_streaming_inference_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_streaming_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_streaming_lti_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_streaming_lti_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_azure_streaming_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_azure_streaming_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_payload_inference_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_payload_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_payload_lti_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_payload_lti_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_payload_only_load_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_payload_only_load_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_payload_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_payload_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_streaming_inference_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_streaming_inference_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_streaming_lti_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_streaming_lti_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_streaming_only_load_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_streaming_only_load_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_streaming_payload_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_streaming_payload_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)


@pytest.mark.benchmark
def test_dfp_modules_duo_streaming_training_e2e(benchmark: typing.Any):

    pipe_conf = PIPELINES_CONF.get("test_dfp_modules_duo_streaming_training_e2e")

    bcg = BenchmarkConfGenerator(pipe_conf=pipe_conf, tracking_uri=TRACKING_URI)

    pipe_config = bcg.pipe_config
    module_config = bcg.get_module_conf()
    input_filenames = bcg.get_filenames()

    benchmark(dfp_modules_pipeline, pipe_config, module_config, filenames=input_filenames)
