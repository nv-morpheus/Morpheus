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

import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.type_aliases import DataFrameType


def _check_pass_thru(config: Config,
                     filter_probs_df: DataFrameType,
                     pass_thru_stage_cls: SinglePortStage,
                     on_data_fn_name: str = 'on_data'):
    stage = pass_thru_stage_cls(config)
    assert isinstance(stage, SinglePortStage)

    meta = MessageMeta(filter_probs_df)
    multi = MultiMessage(meta=meta)

    on_data_fn = getattr(stage, on_data_fn_name)
    assert on_data_fn(multi) is multi


@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru.py'))
def test_pass_thru_ex1(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.PassThruStage)


@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru_deco.py'))
def test_pass_thru_ex1_deco(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.pass_thru_stage, on_data_fn_name='_on_data_fn')


@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/3_simple_cpp_stage/pass_thru.py'))
def test_pass_thru_ex3(config: Config, filter_probs_df: DataFrameType, import_mod: types.ModuleType):
    pass_thru = import_mod
    _check_pass_thru(config, filter_probs_df, pass_thru.PassThruStage)
