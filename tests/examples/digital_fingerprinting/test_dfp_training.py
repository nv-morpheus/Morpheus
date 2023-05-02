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
def test_on_data(
        config: Config,
        dfp_multi_message: "MultiDFPMessage",  # noqa: F821
        dataset_pandas: DatasetManager,
        validation_size: float):
    from dfp.stages.dfp_training import DFPTraining
    config.ae.feature_columns = ['v2', 'v3']
    expected_df = dfp_multi_message.get_meta_dataframe().copy(deep=True)

    stage = DFPTraining(config)
    results = stage.on_data(dfp_multi_message)

    assert isinstance(results, MultiAEMessage)
    dataset_pandas.assert_df_equal(results.get_meta(), expected_df)
