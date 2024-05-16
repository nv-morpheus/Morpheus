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
import pandas as pd
import pytest

import morpheus._lib.messages as _messages
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import MultiResponseAEMessage
from morpheus.messages import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.stages.postprocess.timeseries_stage import TimeSeriesStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.feature_length = 256
    config.ae = ConfigAutoEncoder()
    config.ae.feature_columns = ["data"]
    config.ae.timestamp_column_name = "ts"
    yield config


def _make_multi_response_ae_message(df, probs):
    df_ = df[0:len(probs)]
    mem = ResponseMemory(count=len(df_), tensors={'probs': probs})

    return MultiResponseAEMessage(meta=MessageMeta(df_), count=len(df_), memory=mem, user_id="test_user_id")


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(_messages.TensorMemory(count=len(df_), tensors={'probs': probs}))
    cm.set_metadata("user_id", "test_user_id")

    return cm


def test_constructor(config):
    stage = TimeSeriesStage(config)
    assert stage.name == "timeseries"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


@pytest.mark.use_cudf
@pytest.mark.use_python
def test_call_timeseries_user(config):
    stage = TimeSeriesStage(config)

    df = pd.DataFrame({"ts": pd.date_range(start='01-01-2022', periods=5)})
    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_multi_response_ae_message = _make_multi_response_ae_message(df, probs)
    mock_control_message = _make_control_message(df, probs)

    assert stage._call_timeseries_user(mock_multi_response_ae_message)[0].user_id == "test_user_id"
    assert stage._call_timeseries_user(mock_control_message)[0].get_metadata("user_id") == "test_user_id"
