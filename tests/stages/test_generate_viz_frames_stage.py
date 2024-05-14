# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import Mock
from unittest.mock import patch

import cupy as cp
from networkx import moebius_kantor_graph
import pytest

import cudf
import pandas as pd

from morpheus._lib.messages import ResponseMemory, TensorMemory
from morpheus.config import Config, ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiResponseMessage
from morpheus.stages.postprocess.generate_viz_frames_stage import GenerateVizFramesStage
import morpheus._lib.messages as _messages


def _make_multi_response_message(df, probs):
    df_ = df[0:len(probs)]
    mem = ResponseMemory(count=len(df_), tensors={'probs': probs})

    return MultiResponseMessage(meta=MessageMeta(df_), memory=mem)


def _make_control_message(df, probs):
    df_ = df[0:len(probs)]
    cm = ControlMessage()
    cm.payload(MessageMeta(df_))
    cm.tensors(_messages.TensorMemory(count=len(df_), tensors={'probs': probs}))

    return cm


def test_constructor(config: Config):
    stage = GenerateVizFramesStage(config)
    assert stage.name == "gen_viz"

    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


def test_process_control_message_and_multi_message(config: Config):
    stage = GenerateVizFramesStage(config)

    df = cudf.DataFrame({
        "timestamp": [1616380971990, 1616380971991],
        "src_ip": ["10.20.16.248", "10.244.0.1"],
        "dest_ip": ["10.244.0.59", "10.244.0.25"],
        "src_port": ["50410", "50410"],
        "dest_port": ["80", "80"],
        "data": ["a", "b"]
    })

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_multi_response_message = _make_multi_response_message(df, probs)
    mock_control_message = _make_control_message(df, probs)

    output_multi_response_message_list = stage._to_vis_df(mock_multi_response_message)
    output_control_message_list = stage._to_vis_df(mock_control_message)
    for output_multi_response_message, output_control_message in zip(output_multi_response_message_list, output_control_message_list):
        assert output_multi_response_message[1].equals(output_control_message[1])
