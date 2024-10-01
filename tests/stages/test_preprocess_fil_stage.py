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
from morpheus.config import ConfigFIL
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage


@pytest.fixture(name='config')
def fixture_config(config: Config, use_cpp: bool):  # pylint: disable=unused-argument
    config.feature_length = 1
    config.fil = ConfigFIL()
    config.fil.feature_columns = ["data"]
    yield config


def test_constructor(config: Config):
    stage = PreprocessFILStage(config)
    assert stage.name == "preprocess-fil"
    assert stage._fea_length == config.feature_length
    assert stage.features == config.fil.feature_columns

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


def test_process_control_message(config: Config):
    stage = PreprocessFILStage(config)
    input_cm = ControlMessage()
    df = cudf.DataFrame({"data": [1, 2, 3]})
    meta = MessageMeta(df)
    input_cm.payload(meta)

    output_cm = stage.pre_process_batch(input_cm, stage._fea_length, stage.features)
    assert cp.array_equal(output_cm.tensors().get_tensor("input__0"), cp.asarray(df.to_cupy()))
    expect_seq_ids = cp.zeros((df.shape[0], 3), dtype=cp.uint32)
    expect_seq_ids[:, 0] = cp.arange(0, df.shape[0], dtype=cp.uint32)
    expect_seq_ids[:, 2] = stage._fea_length - 1
    assert cp.array_equal(output_cm.tensors().get_tensor("seq_ids"), expect_seq_ids)
