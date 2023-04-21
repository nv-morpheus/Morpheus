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
import types
import typing

import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import AppShieldMessageMeta
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.stages.input.appshield_source_stage import AppShieldSourceStage
from morpheus.stages.preprocess.preprocess_base_stage import PreprocessBaseStage
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


@pytest.mark.use_python
class TestPreprocessingRWStage:

    def test_constructor(self, config: Config, rwd_conf: dict):
        from stages.preprocessing import PreprocessingRWStage

        stage = PreprocessingRWStage(config, feature_columns=rwd_conf['model_features'], sliding_window=6)
        assert stage._feature_columns == rwd_conf['model_features']
        assert stage._features_len == len(rwd_conf['model_features'])
        assert stage._snapshot_dict == {}
        assert len(stage._padding_data) == len(rwd_conf['model_features']) * 6
        for i in stage._padding_data:
            assert i == 0
