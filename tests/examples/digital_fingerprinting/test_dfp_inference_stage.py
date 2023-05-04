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

import glob
import os
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from utils import TEST_DIRS


@pytest.fixture(autouse=True)
def mock_mlflow_client():
    with mock.patch("dfp.stages.dfp_inference_stage.MlflowClient") as mock_mlflow_client:
        mock_mlflow_client.return_value = mock_mlflow_client
        yield mock_mlflow_client


@pytest.fixture(autouse=True)
def mock_model_manager():
    with mock.patch("dfp.stages.dfp_inference_stage.ModelManager") as mock_model_manager:
        mock_model_manager.return_value = mock_model_manager
        yield mock_model_manager


def test_constructor(config: Config, mock_mlflow_client: mock.MagicMock, mock_model_manager: mock.MagicMock):
    from dfp.stages.dfp_inference_stage import DFPInferenceStage

    stage = DFPInferenceStage(config, model_name_formatter="test_model_name-{user_id}-{user_md5}")

    assert isinstance(stage, SinglePortStage)
    assert stage._client is mock_mlflow_client
    assert stage._fallback_user == config.ae.fallback_username
    assert stage._model_cache == {}
    assert stage._model_manager is mock_model_manager

    mock_mlflow_client.assert_called_once()
    mock_model_manager.assert_called_once_with(model_name_formatter="test_model_name-{user_id}-{user_md5}")
