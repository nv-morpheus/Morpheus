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


def test_constructor(config: Config):
    from dfp.stages.dfp_mlflow_model_writer import DFPMLFlowModelWriterStage

    stage = DFPMLFlowModelWriterStage(
        config,
        model_name_formatter="test_model_name-{user_id}-{user_md5}",
        experiment_name_formatter="test_experiment_name-{user_id}-{user_md5}-{reg_model_name}",
        databricks_permissions={'test': 'this'})
    assert isinstance(stage, SinglePortStage)
    assert stage._model_name_formatter == "test_model_name-{user_id}-{user_md5}"
    assert stage._experiment_name_formatter == "test_experiment_name-{user_id}-{user_md5}-{reg_model_name}"
    assert stage._databricks_permissions == {'test': 'this'}
