#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from morpheus.config import AEFeatureScalar
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_response_message import MultiResponseProbsMessage
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.auto_encoder_inference_stage import AutoEncoderInferenceStage
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from morpheus.stages.preprocess import preprocess_ae_stage
from morpheus.stages.preprocess import train_ae_stage
from utils import TEST_DIRS
from utils import calc_error_val

# End-to-end test intended to imitate the DFP validation test


@pytest.mark.slow
@pytest.mark.use_python
def test_dfp_roleg(config, tmp_path):
    config.mode = PipelineModes.AE
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.num_threads = 1
    config.use_cpp = False

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "role-g"
    config.ae.feature_scaler = AEFeatureScalar.STANDARD

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-*-input.csv")
    train_data_glob = os.path.join(TEST_DIRS.training_data_dir, "dfp-cloudtrail-*.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-role-g-validation-data-output.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(
        train_ae_stage.TrainAEStage(
            config,
            train_data_glob=train_data_glob,
            source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
            seed=42,
            sort_glob=True))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        index_col="_index_",
                        include=["mean_abs_z", "max_abs_z", ".*_z_loss", ".*_loss", ".*_pred"],
                        rel_tol=0.15,
                        overwrite=True))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 0


@pytest.mark.slow
@pytest.mark.use_python
def test_dfp_user123(config, tmp_path):
    config.mode = PipelineModes.AE
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.num_threads = 1
    config.use_cpp = False

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "user123"
    config.ae.feature_scaler = "standard"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-*-input.csv")
    train_data_glob = os.path.join(TEST_DIRS.training_data_dir, "dfp-cloudtrail-*.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-user123-validation-data-output.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(
        train_ae_stage.TrainAEStage(
            config,
            train_data_glob=train_data_glob,
            source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
            seed=42,
            sort_glob=True))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        index_col="_index_",
                        include=["mean_abs_z", "max_abs_z", ".*_loss", ".*_z_loss"],
                        rel_tol=0.1))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 0


@pytest.mark.slow
@pytest.mark.use_python
def test_dfp_user123_multi_segment(config, tmp_path):
    config.mode = PipelineModes.AE
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.num_threads = 1
    config.use_cpp = False

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "user123"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-*-input.csv")
    train_data_glob = os.path.join(TEST_DIRS.training_data_dir, "dfp-cloudtrail-*.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-user123-validation-data-output.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_segment_boundary(UserMessageMeta)  # Boundary 1
    pipe.add_stage(
        train_ae_stage.TrainAEStage(
            config,
            train_data_glob=train_data_glob,
            source_stage_class="morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage",
            seed=42,
            sort_glob=True))
    pipe.add_segment_boundary(MultiAEMessage)  # Boundary 2
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_segment_boundary(MultiAEMessage)  # Boundary 3
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        index_col="_index_",
                        include=["mean_abs_z", "max_abs_z", ".*_loss", ".*_z_loss"],
                        rel_tol=0.1))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_segment_boundary(MessageMeta)  # Boundary 4
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 0
