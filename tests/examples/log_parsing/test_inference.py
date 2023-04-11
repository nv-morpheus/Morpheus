#!/usr/bin/env python
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

import os
import typing
from unittest import mock

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceMessage
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue
from utils import TEST_DIRS


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_triton_inference_log_parsing_constructor(config: Config, import_mod: typing.List[typing.Any]):
    inference_mod = import_mod[0]
    pq = ProducerConsumerQueue()
    worker = inference_mod.TritonInferenceLogParsing(inf_queue=pq,
                                                     c=config,
                                                     model_name='test_model',
                                                     server_url='test_server',
                                                     force_convert_inputs=False,
                                                     use_shared_memory=False,
                                                     inout_mapping={'test': 'this'})

    assert worker._model_name == 'test_model'
    assert worker._server_url == 'test_server'
    assert not worker._force_convert_inputs
    assert not worker._use_shared_memory

    expected_mapping = inference_mod.TritonInferenceLogParsing.default_inout_mapping()
    expected_mapping.update({'test': 'this'})
    assert worker._inout_mapping == expected_mapping


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
@pytest.mark.parametrize("mess_offset,mess_count,offset,count", [(0, 20, 0, 20), (5, 10, 5, 10)])
def test_log_parsing_triton_inference_log_parsing_build_output_message(config: Config,
                                                                       filter_probs_df: typing.Union[pd.DataFrame,
                                                                                                     cudf.DataFrame],
                                                                       import_mod: typing.List[typing.Any],
                                                                       mess_offset,
                                                                       mess_count,
                                                                       offset,
                                                                       count):
    inference_mod = import_mod[0]
    tensor_length = offset + count
    seq_ids = cp.zeros((tensor_length, 3), dtype=cp.uint32)
    seq_ids[offset:offset + count, 0] = cp.arange(mess_offset, mess_offset + count, dtype=cp.uint32)
    seq_ids[:, 2] = 42

    meta = MessageMeta(filter_probs_df)
    input_mem = InferenceMemoryNLP(count=tensor_length,
                                   input_ids=cp.zeros((tensor_length, 2), dtype=cp.float32),
                                   input_mask=cp.ones((tensor_length, 2), dtype=cp.float32),
                                   seq_ids=seq_ids)

    input_msg = MultiInferenceMessage(meta=meta,
                                      mess_offset=mess_offset,
                                      mess_count=mess_count,
                                      memory=input_mem,
                                      offset=offset,
                                      count=count)

    worker = inference_mod.TritonInferenceLogParsing(inf_queue=ProducerConsumerQueue(),
                                                     c=config,
                                                     model_name='test_model',
                                                     server_url='test_server',
                                                     force_convert_inputs=False,
                                                     use_shared_memory=False)

    mock_inout = mock.MagicMock()
    mock_inout.shape = (count, 2)
    worker._inputs['test'] = mock_inout

    msg = worker.build_output_message(input_msg)
    assert msg.meta is meta
    assert msg.mess_offset == mess_offset
    assert msg.mess_count == count
    assert msg.offset == 0
    assert msg.count == count

    assert set(msg.memory.tensor_names).issuperset(('confidences', 'labels', 'input_ids', 'seq_ids'))
    assert msg.confidences.shape == (count, 2)
    assert msg.labels.shape == (count, 2)
    assert msg.input_ids.shape == (count, 2)
    assert msg.seq_ids.shape == (count, 3)


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_inference_stage(config: Config, import_mod: typing.List[typing.Any]):
    inference_mod = import_mod[0]
    config.mode = PipelineModes.NLP
    """
    log_example_dir = os.path.join(TEST_DIRS.examples_dir, 'log_parsing')
    log_test_data_dir = os.path.join(TEST_DIRS.tests_data_dir, 'log_parsing')

    sys.path.append(log_example_dir)
    mod_path = os.path.join(log_example_dir, 'postprocessing.py')
    LogParsingPostProcessingStage = get_plugin_stage_class(mod_path, "log-postprocess", mode=config.mode)

    model_vocab_file = os.path.join(TEST_DIRS.models_dir,
                                    'training-tuning-scripts/sid-models/resources/bert-base-cased-vocab.txt')
    model_config_file = os.path.join(TEST_DIRS.tests_data_dir, 'log-parsing-config.json')

    stage = LogParsingPostProcessingStage(config, vocab_path=model_vocab_file, model_config_path=model_config_file)

    post_proc_message = build_post_proc_message(log_example_dir, log_test_data_dir)
    expected_df = read_file_to_df(os.path.join(log_test_data_dir, 'expected_out.csv'), df_type='pandas')

    out_meta = stage._postprocess(post_proc_message)

    assert isinstance(out_meta, MessageMeta)
    assert_df_equal(out_meta._df, expected_df)
    """
