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

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from utils.dataset_manager import DatasetManager


@pytest.fixture
def dfp_multi_message(config, dfp_multi_message):
    # Fill in some values for columns that the stage is looking for
    with dfp_multi_message.meta.mutable_dataframe() as df:
        step = (len(df) + 1) * 100
        df["mean_abs_z"] = [i for i in range(0, len(df) * step, step)]
        for (i, col) in enumerate(sorted(config.ae.feature_columns)):
            step = i + 1 * 100
            df[f"{col}_z_loss"] = [k for k in range(0, len(df) * step, step)]

    yield dfp_multi_message


def test_constructor(config: Config):
    from dfp.stages.dfp_viz_postproc import DFPVizPostprocStage
    stage = DFPVizPostprocStage(config, period='M', output_dir='/fake/test/dir', output_prefix='test_prefix')

    assert isinstance(stage, SinglePortStage)
    assert stage._user_column_name == config.ae.userid_column_name
    assert stage._timestamp_column == config.ae.timestamp_column_name
    assert stage._feature_columns == config.ae.feature_columns
    assert stage._period == 'M'
    assert stage._output_dir == '/fake/test/dir'
    assert stage._output_prefix == 'test_prefix'
    assert stage._output_filenames == []


def test_postprocess(
        config: Config,
        dfp_multi_message: "MultiDFPMessage",  # noqa: F821
        dataset_pandas: DatasetManager):
    from dfp.stages.dfp_viz_postproc import DFPVizPostprocStage

    df = dfp_multi_message.meta.copy_dataframe()
    expected_df = pd.DataFrame()
    expected_df["user"] = df[config.ae.userid_column_name]
    expected_df["time"] = df[config.ae.timestamp_column_name]
    expected_df["period"] = pd.to_datetime(df[config.ae.timestamp_column_name]).dt.to_period('M')
    expected_df["anomalyScore"] = df['mean_abs_z']
    for col in config.ae.feature_columns:
        expected_df[f"{col}_score"] = df[f"{col}_z_loss"]

    # _postprocess doesn't write to disk, so the fake output_dir, shouldn't be an issue
    stage = DFPVizPostprocStage(config, period='M', output_dir='/fake/test/dir', output_prefix='test_prefix')
    results = stage._postprocess(dfp_multi_message)

    assert isinstance(results, MessageMeta)
    dataset_pandas.assert_compare_df(results.copy_dataframe(), expected_df)
