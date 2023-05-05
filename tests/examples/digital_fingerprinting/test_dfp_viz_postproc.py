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
from unittest import mock

import numpy as np
import pytest

from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.logger import set_log_level


def test_constructor(config: Config):
    from dfp.stages.dfp_viz_postproc import DFPVizPostprocStage
    stage = DFPVizPostprocStage(config, period='M', output_dir='/test/dir', output_prefix='test_prefix')

    assert isinstance(stage, SinglePortStage)
    assert stage._user_column_name == config.ae.userid_column_name
    assert stage._timestamp_column == config.ae.timestamp_column_name
    assert stage._feature_columns == config.ae.feature_columns
    assert stage._period == 'M'
    assert stage._output_dir == '/test/dir'
    assert stage._output_prefix == 'test_prefix'
    assert stage._output_filenames == []
