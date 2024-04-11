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

from morpheus._lib.messages import TensorMemory
from morpheus.config import Config, ConfigAutoEncoder
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.messages import MultiResponseMessage
from morpheus.stages.postprocess.generate_viz_frames_stage import GenerateVizFramesStage


@pytest.fixture(name='config')
def fixture_config(config: Config):
    # config.feature_length = 256
    # config.ae = ConfigAutoEncoder()
    # config.ae.feature_columns = ["data"]
    yield config


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
    meta = MessageMeta(df)

    memory = TensorMemory(count=1, tensors=None)
    input_multi_resp_message = MultiResponseMessage(meta=meta,
                                                    mess_offset=0,
                                                    mess_count=1,
                                                    memory=memory,
                                     offset=0,
                                                    count=1,
                                                    id_tensor_name="seq_ids",
                                                    probs_tensor_name="probs")

    output_list = stage._to_vis_df(input_multi_resp_message)
    print("-----------result-----------")
    print(output_list)

    # TODO(Yuchen): Check if the output message has identical tensors after supporting ControlMessage

    # Check if each tensor in the control message is equal to the corresponding tensor in the inference message
    # for tensor_key in output_control_message.tensors().tensor_names:
    #     assert cp.array_equal(output_control_message.tensors().get_tensor(tensor_key),
    #                           getattr(output_infer_message, tensor_key))
