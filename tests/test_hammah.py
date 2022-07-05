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
import pytest

from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
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

# End-to-end test intended to imitate the hammah validation test


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.reload_modules(preprocess_ae_stage)
@pytest.mark.reload_modules(train_ae_stage)
@pytest.mark.usefixtures("reload_modules")
@mock.patch('morpheus.stages.preprocess.train_ae_stage.AutoEncoder')
def test_hammah_roleg(mock_ae, config, tmp_path):
    tensor_data = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'hammah_roleg_tensor.csv'), delimiter=',')
    anomaly_score = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'hammah_roleg_anomaly_score.csv'), delimiter=',')

    mock_input_tesnsor = mock.MagicMock()
    mock_input_tesnsor.return_value = mock_input_tesnsor
    mock_input_tesnsor.detach.return_value = tensor_data

    mock_ae.return_value = mock_ae
    mock_ae.build_input_tensor.return_value = mock_input_tesnsor
    mock_ae.get_anomaly_score.return_value = anomaly_score

    config.mode = PipelineModes.AE
    config.class_labels = ["ae_anomaly_score"]
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.feature_length = 256
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "role-g"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "hammah-*.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'hammah-role-g-validation-data.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(train_ae_stage.TrainAEStage(config, train_data_glob=input_glob, seed=42, sort_glob=True))
    pipe.add_stage(preprocess_ae_stage.PreprocessAEStage(config))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(
        TimeSeriesStage(config,
                        resolution="10m",
                        min_window="12 h",
                        hot_start=False,
                        cold_end=False,
                        filter_percent=90.0,
                        zscore_threshold=8.0))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        index_col="_index_",
                        exclude=("event_dt", ),
                        rel_tol=0.15))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    mock_ae.fit.assert_called_once()
    mock_ae.build_input_tensor.assert_called_once()
    mock_ae.get_anomaly_score.assert_called_once()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 3


@pytest.mark.slow
@pytest.mark.use_python
@pytest.mark.reload_modules(preprocess_ae_stage)
@pytest.mark.reload_modules(train_ae_stage)
@pytest.mark.usefixtures("reload_modules")
@mock.patch('morpheus.stages.preprocess.train_ae_stage.AutoEncoder')
def test_hammah_user123(mock_ae, config, tmp_path):
    tensor_data = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'hammah_user123_tensor.csv'), delimiter=',')
    anomaly_score = np.loadtxt(os.path.join(TEST_DIRS.tests_data_dir, 'hammah_user123_anomaly_score.csv'),
                               delimiter=',')

    mock_input_tesnsor = mock.MagicMock()
    mock_input_tesnsor.return_value = mock_input_tesnsor
    mock_input_tesnsor.detach.return_value = tensor_data

    mock_ae.return_value = mock_ae
    mock_ae.build_input_tensor.return_value = mock_input_tesnsor
    mock_ae.get_anomaly_score.return_value = anomaly_score

    config.mode = PipelineModes.AE
    config.class_labels = ["ae_anomaly_score"]
    config.model_max_batch_size = 1024
    config.pipeline_batch_size = 1024
    config.edge_buffer_size = 128
    config.num_threads = 1

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = "userIdentitysessionContextsessionIssueruserName"
    config.ae.userid_filter = "user123"

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    input_glob = os.path.join(TEST_DIRS.validation_data_dir, "hammah-*.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'hammah-user123-validation-data.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(CloudTrailSourceStage(config, input_glob=input_glob, sort_glob=True))
    pipe.add_stage(train_ae_stage.TrainAEStage(config, train_data_glob=input_glob, seed=42, sort_glob=True))
    pipe.add_stage(preprocess_ae_stage.PreprocessAEStage(config))
    pipe.add_stage(AutoEncoderInferenceStage(config))
    pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(
        TimeSeriesStage(config,
                        resolution="1m",
                        min_window="12 h",
                        hot_start=True,
                        cold_end=False,
                        filter_percent=90.0,
                        zscore_threshold=8.0))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(
        ValidationStage(config,
                        val_file_name=val_file_name,
                        results_file_name=results_file_name,
                        index_col="_index_",
                        exclude=("event_dt", ),
                        rel_tol=0.1))
    pipe.add_stage(SerializeStage(config, include=[]))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()

    mock_ae.fit.assert_called_once()
    mock_ae.build_input_tensor.assert_called_once()
    mock_ae.get_anomaly_score.assert_called_once()

    results = calc_error_val(results_file_name)
    assert results.diff_rows == 48
