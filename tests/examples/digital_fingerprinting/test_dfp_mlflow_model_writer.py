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
import os
from collections import OrderedDict
from unittest import mock

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.models.dfencoder import AutoEncoder
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.logger import set_log_level
from utils import TEST_DIRS
from utils import import_or_skip
from utils.dataset_manager import DatasetManager


@pytest.fixture(autouse=True, scope='session')
def mlflow(fail_missing: bool):
    """
    Mark tests requiring mlflow
    """
    yield import_or_skip("mlflow", reason="dfp_mlflow_model_writer tests require mlflow ", fail_missing=fail_missing)


@pytest.fixture
def databricks_env(restore_environ):
    env = {'DATABRICKS_HOST': 'https://test_host', 'DATABRICKS_TOKEN': 'test_token'}
    os.environ.update(env)
    yield env


def test_constructor(config: Config):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config,
                                      model_name_formatter="test_model_name-{user_id}-{user_md5}",
                                      experiment_name_formatter="/test/{user_id}-{user_md5}-{reg_model_name}",
                                      databricks_permissions={'test': 'this'})
    assert isinstance(stage, SinglePortStage)
    assert stage._model_name_formatter == "test_model_name-{user_id}-{user_md5}"
    assert stage._experiment_name_formatter == "/test/{user_id}-{user_md5}-{reg_model_name}"
    assert stage._databricks_permissions == {'test': 'this'}


@pytest.mark.parametrize(
    "model_name_formatter,user_id,expected_val",
    [("test_model_name-{user_id}", 'test_user', "test_model_name-test_user"),
     ("test_model_name-{user_id}-{user_md5}", 'test_user',
      "test_model_name-test_user-9da1f8e0aecc9d868bad115129706a77"),
     ("test_model_name-{user_id}", 'test_城安宮川', "test_model_name-test_城安宮川"),
     ("test_model_name-{user_id}-{user_md5}", 'test_城安宮川', "test_model_name-test_城安宮川-c9acc3dec97777c8b6fd8ae70a744ea8")
     ])
def test_user_id_to_model(config: Config, model_name_formatter: str, user_id: str, expected_val: str):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config, model_name_formatter=model_name_formatter)
    assert stage.user_id_to_model(user_id) == expected_val


@pytest.mark.parametrize("experiment_name_formatter,user_id,expected_val",
                         [("/test/expr/{reg_model_name}", 'test_user', "/test/expr/dfp-test_user"),
                          ("/test/expr/{reg_model_name}-{user_id}", 'test_user', "/test/expr/dfp-test_user-test_user"),
                          ("/test/expr/{reg_model_name}-{user_id}-{user_md5}",
                           'test_user',
                           "/test/expr/dfp-test_user-test_user-9da1f8e0aecc9d868bad115129706a77"),
                          ("/test/expr/{reg_model_name}", 'test_城安宮川', "/test/expr/dfp-test_城安宮川"),
                          ("/test/expr/{reg_model_name}-{user_id}", 'test_城安宮川', "/test/expr/dfp-test_城安宮川-test_城安宮川"),
                          ("/test/expr/{reg_model_name}-{user_id}-{user_md5}",
                           'test_城安宮川',
                           "/test/expr/dfp-test_城安宮川-test_城安宮川-c9acc3dec97777c8b6fd8ae70a744ea8")])
def test_user_id_to_experiment(config: Config, experiment_name_formatter: str, user_id: str, expected_val: str):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(config,
                                      model_name_formatter="dfp-{user_id}",
                                      experiment_name_formatter=experiment_name_formatter)
    assert stage.user_id_to_experiment(user_id) == expected_val


@mock.patch("requests.patch")
@mock.patch("requests.get")
def test_apply_model_permissions(mock_requests_get: mock.MagicMock,
                                 mock_requests_patch: mock.MagicMock,
                                 config: Config,
                                 databricks_env: dict):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    mock_response = mock.MagicMock(status_code=200)
    mock_response.json.return_value = {'registered_model_databricks': {'id': 'test_id'}}
    mock_requests_get.return_value = mock_response

    stage = DFPMLFlowModelWriterStage(config,
                                      databricks_permissions=OrderedDict([('group1', 'CAN_READ'),
                                                                          ('group2', 'CAN_WRITE')]))
    stage._apply_model_permissions("test_experiment")

    expected_headers = {"Authorization": "Bearer {DATABRICKS_TOKEN}".format(**databricks_env)}
    mock_requests_get.assert_called_once_with(
        url="{DATABRICKS_HOST}/api/2.0/mlflow/databricks/registered-models/get".format(**databricks_env),
        headers=expected_headers,
        params={"name": "test_experiment"})

    # get(url='/api/2.0/mlflow/databricks/registered-models/get', headers={'Authorization': 'Bearer test_token'}, params={'name': 'test_experiment'})

    mock_requests_patch.assert_called_once_with(
        url="{DATABRICKS_HOST}/api/2.0/preview/permissions/registered-models/test_id".format(**databricks_env),
        headers=expected_headers,
        json={
            'access_control_list': [{
                'group_name': 'group1', 'permission_level': 'CAN_READ'
            }, {
                'group_name': 'group2', 'permission_level': 'CAN_WRITE'
            }]
        })


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize("databricks_host,databricks_token", [
    ("test_host", None),
    (None, "test_token"),
    (None, None),
])
@mock.patch("requests.patch")
@mock.patch("requests.get")
def test_apply_model_permissions_no_perms_error(mock_requests_get: mock.MagicMock,
                                                mock_requests_patch: mock.MagicMock,
                                                config: Config,
                                                databricks_host: str,
                                                databricks_token: str):
    if databricks_host is not None:
        os.environ["DATABRICKS_HOST"] = databricks_host
    else:
        os.environ.pop("DATABRICKS_HOST", None)

    if databricks_token is not None:
        os.environ["DATABRICKS_TOKEN"] = databricks_token
    else:
        os.environ.pop("DATABRICKS_TOKEN", None)

    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    stage = DFPMLFlowModelWriterStage(config)
    with pytest.raises(RuntimeError):
        stage._apply_model_permissions("test_experiment")

    mock_requests_get.assert_not_called()
    mock_requests_patch.assert_not_called()


@pytest.mark.usefixtures("databricks_env")
@mock.patch("requests.patch")
@mock.patch("requests.get")
def test_apply_model_permissions_requests_error(mock_requests_get: mock.MagicMock,
                                                mock_requests_patch: mock.MagicMock,
                                                config: Config):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage
    mock_requests_get.side_effect = RuntimeError("test error")

    stage = DFPMLFlowModelWriterStage(config)
    stage._apply_model_permissions("test_experiment")

    # This method just catches and logs any errors
    mock_requests_get.assert_called_once()
    mock_requests_patch.assert_not_called()
