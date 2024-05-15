#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import pytest
import pandas as pd

from morpheus.common import FilterSource
import morpheus._lib.messages as _messages
from morpheus.messages import MultiMessage, ControlMessage
from morpheus.messages import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.validation_stage import ValidationStage


def _make_multi_message(df):
    return MultiMessage(meta=MessageMeta(df))


def _make_control_message(df):
    cm = ControlMessage()
    cm.payload(MessageMeta(df))

    return cm


def test_constructor(config):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    stage = ValidationStage(config, val_file_name=df)
    assert stage.name == "validation"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


def test_do_comparison(config):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    stage = ValidationStage(config, val_file_name=df)

    mm = _make_multi_message(df)
    cm = _make_control_message(df)

    stage._append_message(mm)
    mm_results = stage.get_results(clear=True)
    stage._append_message(cm)
    cm_results = stage.get_results(clear=True)
    assert mm_results == cm_results
