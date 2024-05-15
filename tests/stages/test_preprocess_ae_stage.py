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
from morpheus.config import ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiAEMessage
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

    accepted_types = stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


def test_process_control_message_and_multi_message(config: Config):
    stage = PreprocessAEStage(config)

    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)

    input_multi_ae_message = MultiAEMessage(meta=meta,
                                            mess_offset=0,
                                            mess_count=3,
                                            model=None,
                                            train_scores_mean=0.0,
                                            train_scores_std=1.0)

    output_multi_inference_ae_message = stage.pre_process_batch(input_multi_ae_message,
                                                                fea_len=256,
                                                                feature_columns=["data"])

    input_control_message = ControlMessage()
    input_control_message.payload(meta)

    output_control_message = stage.pre_process_batch(input_control_message, fea_len=256, feature_columns=["data"])

    # Check if each tensor in the control message is equal to the corresponding tensor in the inference message
    for tensor_key in output_control_message.tensors().tensor_names:
        assert cp.array_equal(output_control_message.tensors().get_tensor(tensor_key),
                              getattr(output_multi_inference_ae_message, tensor_key))
