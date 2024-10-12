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

import typing

import cupy as cp
import pytest
import typing_utils

import cudf

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.preprocess.preprocess_ae_stage import PreprocessAEStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.feature_length = 256
    config.ae = ConfigAutoEncoder()
    config.ae.feature_columns = ["data"]
    yield config


def test_constructor(config: Config):
    stage = PreprocessAEStage(config)
    assert stage.name == "preprocess-ae"

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


def test_process_control_message(config: Config):
    stage = PreprocessAEStage(config)

    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)

    input_control_message = ControlMessage()
    input_control_message.payload(meta)

    output_control_message = stage.pre_process_batch(input_control_message, fea_len=256, feature_columns=["data"])

    expected_input = cp.zeros(df.shape, dtype=cp.float32)
    assert cp.array_equal(output_control_message.tensors().get_tensor("input"), expected_input)

    expect_seq_ids = cp.zeros((df.shape[0], 3), dtype=cp.uint32)
    expect_seq_ids[:, 0] = cp.arange(0, df.shape[0], dtype=cp.uint32)
    expect_seq_ids[:, 2] = stage._fea_length - 1
    assert cp.array_equal(output_control_message.tensors().get_tensor("seq_ids"), expect_seq_ids)
