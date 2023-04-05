#!/usr/bin/env python
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
import sys
import typing

import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from utils import TEST_DIRS
from utils import get_plugin_stage_class


@pytest.mark.parametrize(
    'mod_path,set_python_path',
    [(os.path.join(TEST_DIRS.examples_dir, 'developer_guide/1_simple_python_stage/pass_thru.py'), False),
     (os.path.join(TEST_DIRS.examples_dir, 'developer_guide/3_simple_cpp_stage/pass_thru.py'), True)])
@pytest.mark.usefixtures("reset_plugins", "restore_sys_path")
def test_pass_thru(config: Config,
                   filter_probs_df: typing.Union[pd.DataFrame, cudf.DataFrame],
                   mod_path: str,
                   set_python_path: bool):
    if set_python_path:
        sys.path.append(os.path.dirname(mod_path))

    PassThruStage = get_plugin_stage_class(mod_path, "pass-thru", mode=PipelineModes.OTHER)
    stage = PassThruStage(config)

    meta = MessageMeta(filter_probs_df)
    mm = MultiMessage(meta=meta)

    assert stage.on_data(mm) is mm
