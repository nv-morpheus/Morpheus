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
import types
import typing
from io import StringIO

import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from utils import TEST_DIRS


def test_constructor(config: Config, dask_distributed: types.ModuleType, rwd_conf: dict):
    from common.data_models import FeatureConfig
    from common.feature_extractor import FeatureExtractor
    from stages.create_features import CreateFeaturesRWStage

    interested_plugins = ['ldrmodules', 'threadlist', 'envars', 'vadinfo', 'handles']

    n_workers = 12
    threads_per_worker = 8
    stage = CreateFeaturesRWStage(config,
                                  interested_plugins=interested_plugins,
                                  feature_columns=rwd_conf['model_features'],
                                  file_extns=rwd_conf['file_extensions'],
                                  n_workers=n_workers,
                                  threads_per_worker=threads_per_worker)

    assert isinstance(stage, MultiMessageStage)
    assert isinstance(stage._client, dask_distributed.Client)
    scheduler_info = stage._client.scheduler_info()
    len(scheduler_info['workers']) == n_workers
    for worker in scheduler_info['workers'].values():
        assert worker['nthreads'] == threads_per_worker

    assert isinstance(stage._feature_config, FeatureConfig)
    assert stage._feature_config.file_extns == rwd_conf['file_extensions']
    assert stage._feature_config.interested_plugins == interested_plugins

    assert stage._feas_all_zeros == {c: 0 for c in rwd_conf['model_features']}

    assert isinstance(stage._fe, FeatureExtractor)
    assert stage._fe._config is stage._feature_config
