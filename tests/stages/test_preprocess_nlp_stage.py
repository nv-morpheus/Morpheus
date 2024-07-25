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
from unittest.mock import Mock
from unittest.mock import patch

import cupy as cp
import pytest
import typing_utils

import cudf

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage


@pytest.fixture(name='config')
def fixture_config(config: Config, use_cpp: bool):  # pylint: disable=unused-argument
    config.class_labels = [
        "address",
        "bank_acct",
        "credit_card",
        "email",
        "govt_id",
        "name",
        "password",
        "phone_num",
        "secret_keys",
        "user"
    ]
    config.edge_buffer_size = 4
    config.feature_length = 256
    config.mode = "NLP"
    config.model_max_batch_size = 32
    config.num_threads = 1
    config.pipeline_batch_size = 64
    yield config


def test_constructor(config: Config):
    stage = PreprocessNLPStage(config)
    assert stage.name == "preprocess-nlp"
    assert stage._column == "data"
    assert stage._seq_length == 256
    assert stage._vocab_hash_file.endswith("data/bert-base-cased-hash.txt")
    assert stage._truncation is False
    assert stage._do_lower_case is False
    assert stage._add_special_tokens is False

    accepted_union = typing.Union[stage.accepted_types()]
    assert typing_utils.issubtype(ControlMessage, accepted_union)


@patch("morpheus.stages.preprocess.preprocess_nlp_stage.tokenize_text_series")
def test_process_control_message(mock_tokenize_text_series, config: Config):
    mock_tokenized = Mock()
    mock_tokenized.input_ids = cp.array([[1, 2], [1, 2]])
    mock_tokenized.input_mask = cp.array([[3, 4], [3, 4]])
    mock_tokenized.segment_ids = cp.array([[0, 0], [1, 1]])
    mock_tokenize_text_series.return_value = mock_tokenized

    stage = PreprocessNLPStage(config)
    input_cm = ControlMessage()
    df = cudf.DataFrame({"data": ["a", "b", "c"]})
    meta = MessageMeta(df)
    input_cm.payload(meta)

    output_cm = stage.pre_process_batch(input_cm,
                                        stage._vocab_hash_file,
                                        stage._do_lower_case,
                                        stage._seq_length,
                                        stage._stride,
                                        stage._truncation,
                                        stage._add_special_tokens,
                                        stage._column)
    assert output_cm.get_metadata("inference_memory_params") == {"inference_type": "nlp"}
    assert cp.array_equal(output_cm.tensors().get_tensor("input_ids"), mock_tokenized.input_ids)
    assert cp.array_equal(output_cm.tensors().get_tensor("input_mask"), mock_tokenized.input_mask)
    assert cp.array_equal(output_cm.tensors().get_tensor("seq_ids"), mock_tokenized.segment_ids)
