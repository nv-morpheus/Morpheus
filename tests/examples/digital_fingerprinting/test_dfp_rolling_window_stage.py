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

import json
import os
import typing
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


@pytest.fixture
def dfp_message_meta(config: Config, dataset_pandas: DatasetManager):
    from dfp.messages.multi_dfp_message import DFPMessageMeta

    user_id = 'test_user'
    df = dataset_pandas['filter_probs.csv']
    df[config.ae.timestamp_column_name] = [1683054498 + i for i in range(0, len(df) * 100, 100)]
    df['user_id'] = user_id
    yield DFPMessageMeta(df, user_id)


def test_constructor(config: Config):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')
    assert isinstance(stage, SinglePortStage)
    assert stage._min_history == 5
    assert stage._min_increment == 7
    assert stage._max_history == 100
    assert stage._cache_dir.startswith('/test/path/cache')
    assert stage._user_cache_map == {}


def test_get_user_cache_hit(config: Config):
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = mock.MagicMock()
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
        results2 is results


@pytest.mark.parametrize('use_on_data', [True, False])
def test_build_window_no_new(
        config: Config,
        use_on_data: bool,
        dfp_message_meta: "DFPMessageMeta"  # noqa: F821
):
    from dfp.messages.multi_dfp_message import MultiDFPMessage
    from dfp.stages.dfp_rolling_window_stage import DFPRollingWindowStage

    stage = DFPRollingWindowStage(config, min_history=5, min_increment=7, max_history=100, cache_dir='/test/path/cache')

    mock_cache = mock.MagicMock()
    mock_cache.append_dataframe.return_value = False
    stage._user_cache_map[dfp_message_meta.user_id] = mock_cache

    # on_data is a thin wrapper around _build_window, results should be the same
    if use_on_data:
        results = stage.on_data(dfp_message_meta)
    else:
        results = stage._build_window(dfp_message_meta)

    assert results is None
