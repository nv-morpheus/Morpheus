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

import os
from unittest import mock

import pandas as pd
import pytest

from _utils.dataset_manager import DatasetManager
from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage


def build_mock_user_cache(user_id: str = 'test_user',
                          train_df: pd.DataFrame = None,
                          count: int = 10,
                          total_count: int = 20,
                          last_train_count: int = 10) -> mock.MagicMock:
    mock_cache = mock.MagicMock()
    mock_cache.user_id = user_id
    mock_cache.append_dataframe.return_value = True
    mock_cache.get_train_df.return_value = train_df
    mock_cache.count = count
    mock_cache.total_count = total_count
    mock_cache.last_train_count = last_train_count

    return mock_cache


def test_constructor(config: Config):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')
    assert isinstance(stage, SinglePortStage)
    assert stage._min_history == 5
    assert stage._min_increment == 7
    assert stage._max_history == 100
    assert stage._cache_dir.startswith('/test/path/cache')
    assert not stage._user_cache_map


def test_get_user_cache_hit(config: Config):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = build_mock_user_cache()
    stage._user_cache_map['test_user'] = mock_cache

    with stage._get_user_cache('test_user') as user_cache:
        assert user_cache is mock_cache


def test_get_user_cache_miss(config: Config):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage
    from dfp.utils.cached_user_window import CachedUserWindow

    config.ae.timestamp_column_name = 'test_timestamp_col'
    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    with stage._get_user_cache('test_user') as results:
        assert isinstance(results, CachedUserWindow)
        assert results.user_id == 'test_user'
        assert results.cache_location == os.path.join(stage._cache_dir, 'test_user.pkl')
        assert results.timestamp_column == 'test_timestamp_col'

    with stage._get_user_cache('test_user') as results2:
        assert results2 is results


def test_build_window_no_new(
        config: Config,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = build_mock_user_cache()
    mock_cache.append_dataframe.return_value = False
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache
    assert stage._build_window(dfp_message_meta) is None


def test_build_window_not_enough_data(
        config: Config,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = build_mock_user_cache(count=3)
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache
    assert stage._build_window(dfp_message_meta) is None


def test_build_window_min_increment(
        config: Config,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = build_mock_user_cache(count=5, total_count=30, last_train_count=25)
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache
    assert stage._build_window(dfp_message_meta) is None


def test_build_window_invalid(
        config: Config,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    train_df = dfp_message_meta.copy_dataframe()
    # exact values not important so long as they don't match the actual hash
    train_df['_row_hash'] = [-1 for _ in range(len(train_df))]

    mock_cache = build_mock_user_cache(train_df=train_df)
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache

    with pytest.raises(RuntimeError):
        stage._build_window(dfp_message_meta)


def test_build_window_overlap(
        config: Config,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    # Create an overlap
    train_df = dfp_message_meta.copy_dataframe()[-5:]
    train_df['_row_hash'] = pd.util.hash_pandas_object(train_df, index=False)

    mock_cache = build_mock_user_cache(train_df=train_df)
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache

    with pytest.raises(RuntimeError):
        stage._build_window(dfp_message_meta)


@pytest.mark.parametrize('use_on_data', [True, False])
def test_build_window(
        config: Config,
        use_on_data: bool,
        dfp_message_meta: "DFPMessageMeta",  # noqa: F821
        dataset_pandas: DatasetManager):
    from dfp.messages.multi_dfp_message import MultiDFPMessage
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    # Create an overlap
    train_df = dfp_message_meta.copy_dataframe()
    train_df['_row_hash'] = pd.util.hash_pandas_object(train_df, index=False)

    mock_cache = build_mock_user_cache(train_df=train_df)
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache

    # on_data is a thin wrapper around _build_window, results should be the same
    if use_on_data:
        msg = stage.on_data(dfp_message_meta)
    else:
        msg = stage._build_window(dfp_message_meta)

    assert isinstance(msg, MultiDFPMessage)
    assert msg.user_id == dfp_message_meta.user_id
    assert msg.meta.user_id == dfp_message_meta.user_id
    assert msg.mess_offset == 0
    assert msg.mess_count == len(dataset_pandas['filter_probs.csv'])
    dataset_pandas.assert_df_equal(msg.get_meta(), train_df)
    dataset_pandas.assert_df_equal(msg.meta.get_df(), train_df)
    dataset_pandas.assert_df_equal(msg.get_meta_dataframe(), train_df)
