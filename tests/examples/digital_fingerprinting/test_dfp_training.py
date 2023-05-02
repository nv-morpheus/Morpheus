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
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.logger import set_log_level
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


def test_constructor(config: Config):
    from dfp.stages.dfp_training import DFPTraining

    stage = DFPTraining(config, model_kwargs={'test': 'this'}, epochs=40, validation_size=0.5)
    assert isinstance(stage, SinglePortStage)
    assert stage._model_kwargs['test'] == 'this'
    assert stage._epochs == 40
    assert stage._validation_size == 0.5


@pytest.mark.parametrize('validation_size', [-1, -0.2, 1, 5])
def test_constructor_bad_validation_size(config: Config, validation_size: float):
    from dfp.stages.dfp_training import DFPTraining

    with pytest.raises(ValueError):
        stage = DFPTraining(config, validation_size=validation_size)


@pytest.mark.parametrize('validation_size', [0., 0.2])
def test_on_data(config: Config, dataset_pandas: DatasetManager, validation_size: float):
    from dfp.messages.multi_dfp_message import DFPMessageMeta
    from dfp.messages.multi_dfp_message import MultiDFPMessage
    from dfp.stages.dfp_training import DFPTraining

    input_file = os.path.join(TEST_DIRS.validation_data_dir, "dfp-cloudtrail-role-g-validation-data-input.csv")
    df = dataset_pandas[input_file]
    meta = DFPMessageMeta(df, 'Account-123456789')
    msg = MultiDFPMessage(meta=meta)

    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt')) as fh:
        config.ae.feature_columns = [x.strip() for x in fh.readlines()]

    stage = DFPTraining(config)
    results = stage.on_data(msg)

    assert isinstance(results, MultiAEMessage)
    dataset_pandas.assert_compare_df(results.get_meta(), dataset_pandas[input_file])
