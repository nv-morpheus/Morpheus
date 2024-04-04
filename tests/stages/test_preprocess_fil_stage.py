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

import cupy as cp
import pytest

import cudf

from morpheus.config import Config
from morpheus.config import ConfigFIL
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    config.feature_length = 1
    config.fil = ConfigFIL()
    config.fil.feature_columns = ["data"]
    yield config


def test_constructor(config: Config):
    stage = PreprocessFILStage(config)
    assert stage.name == "preprocess-fil"
    assert stage._fea_length == config.feature_length
    assert stage.features == config.fil.feature_columns

    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


def test_process_control_message(config: Config):
    stage = PreprocessFILStage(config)
    input_cm = ControlMessage()
    df = cudf.DataFrame({"data": [1, 2, 3]})
    meta = MessageMeta(df)
    input_cm.payload(meta)

    output_cm = stage.pre_process_batch(input_cm, stage._fea_length, stage.features)
    assert cp.array_equal(output_cm.tensors().get_tensor("input__0"), cp.asarray(df.to_cupy()))
    expect_seg_ids = cp.zeros((df.shape[0], 3), dtype=cp.uint32)
    expect_seg_ids[:, 0] = cp.arange(0, df.shape[0], dtype=cp.uint32)
    expect_seg_ids[:, 2] = stage._fea_length - 1
    assert cp.array_equal(output_cm.tensors().get_tensor("seq_ids"), expect_seg_ids)


def test_process_multi_message(config: Config):
    stage = PreprocessFILStage(config)
    df = cudf.DataFrame({"data": [1, 2, 3]})
    meta = MessageMeta(df)
    mess_offset = 0
    input_multi_message = MultiMessage(meta=meta, mess_offset=mess_offset, mess_count=3)

    output_infer_message = stage.pre_process_batch(input_multi_message, stage._fea_length, stage.features)
    assert cp.array_equal(output_infer_message.input__0, cp.asarray(df.to_cupy()))
    expect_seg_ids = cp.zeros((df.shape[0], 3), dtype=cp.uint32)
    expect_seg_ids[:, 0] = cp.arange(0, df.shape[0], dtype=cp.uint32)
    expect_seg_ids[:, 2] = stage._fea_length - 1
    assert cp.array_equal(output_infer_message.seq_ids, expect_seg_ids)


def test_process_control_message_and_multi_message(config: Config):
    stage = PreprocessFILStage(config)
    df = cudf.DataFrame({"data": [1, 2, 3]})
    meta = MessageMeta(df)
    input_control_message = ControlMessage()
    input_control_message.payload(meta)

    mess_offset = 0
    input_multi_message = MultiMessage(meta=meta, mess_offset=mess_offset, mess_count=3)

    output_control_message = stage.pre_process_batch(input_control_message, stage._fea_length, stage.features)

    output_infer_message = stage.pre_process_batch(input_multi_message, stage._fea_length, stage.features)

    # Check if each tensor in the control message is equal to the corresponding tensor in the inference message
    for tensor_key in output_control_message.tensors().tensor_names:
        assert cp.array_equal(output_control_message.tensors().get_tensor(tensor_key),
                              getattr(output_infer_message, tensor_key))
