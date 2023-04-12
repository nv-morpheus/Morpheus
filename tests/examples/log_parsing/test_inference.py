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
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceMessage
from morpheus.stages.inference.triton_inference_stage import _TritonInferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue
from utils import TEST_DIRS


def build_response_mem(messages_mod, log_test_data_dir: str):
    # we have tensor data for the first five rows
    count = 5
    tensors = {}
    for tensor_name in ['confidences', 'labels']:
        tensor_file = os.path.join(log_test_data_dir, f'{tensor_name}.csv')
        host_data = np.loadtxt(tensor_file, delimiter=',')
        tensors[tensor_name] = cp.asarray(host_data)

    return messages_mod.ResponseMemoryLogParsing(count=count, **tensors)


def build_inf_message(df: typing.Union[pd.DataFrame, cudf.DataFrame],
                      mess_offset: int,
                      mess_count: int,
                      offset: int,
                      count: int,
                      num_cols: int = 2) -> MultiInferenceMessage:
    tensor_length = offset + count
    seq_ids = cp.zeros((tensor_length, 3), dtype=cp.uint32)
    seq_ids[offset:offset + count, 0] = cp.arange(mess_offset, mess_offset + count, dtype=cp.uint32)
    seq_ids[:, 2] = 42

    meta = MessageMeta(df)
    mem = InferenceMemoryNLP(count=tensor_length,
                             input_ids=cp.zeros((tensor_length, num_cols), dtype=cp.float32),
                             input_mask=cp.zeros((tensor_length, num_cols), dtype=cp.float32),
                             seq_ids=seq_ids)

    return MultiInferenceMessage(meta=meta,
                                 mess_offset=mess_offset,
                                 mess_count=mess_count,
                                 memory=mem,
                                 offset=offset,
                                 count=count)


def _check_worker(inference_mod: typing.Any, worker: _TritonInferenceWorker):
    assert isinstance(worker, _TritonInferenceWorker)
    assert isinstance(worker, inference_mod.TritonInferenceLogParsing)
    assert worker._model_name == 'test_model'
    assert worker._server_url == 'test_server'
    assert not worker._force_convert_inputs
    assert not worker._use_shared_memory

    expected_mapping = inference_mod.TritonInferenceLogParsing.default_inout_mapping()
    expected_mapping.update({'test': 'this'})
    assert worker._inout_mapping == expected_mapping


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_triton_inference_log_parsing_constructor(config: Config, import_mod: typing.List[typing.Any]):
    inference_mod = import_mod[0]
    worker = inference_mod.TritonInferenceLogParsing(inf_queue=ProducerConsumerQueue(),
                                                     c=config,
                                                     model_name='test_model',
                                                     server_url='test_server',
                                                     force_convert_inputs=False,
                                                     use_shared_memory=False,
                                                     inout_mapping={'test': 'this'})

    _check_worker(inference_mod, worker)


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
@pytest.mark.parametrize("mess_offset,mess_count,offset,count", [(0, 20, 0, 20), (5, 10, 5, 10)])
def test_log_parsing_triton_inference_log_parsing_build_output_message(config: Config,
                                                                       filter_probs_df: typing.Union[pd.DataFrame,
                                                                                                     cudf.DataFrame],
                                                                       import_mod: typing.List[typing.Any],
                                                                       mess_offset: int,
                                                                       mess_count: int,
                                                                       offset: int,
                                                                       count: int):
    inference_mod = import_mod[0]
    input_msg = build_inf_message(filter_probs_df,
                                  mess_offset=mess_offset,
                                  mess_count=mess_count,
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
    assert msg.meta is input_msg.meta
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
def test_log_parsing_inference_stage_constructor(config: Config, import_mod: typing.List[typing.Any]):
    inference_mod = import_mod[0]
    stage = inference_mod.LogParsingInferenceStage(
        config,
        model_name='test_model',
        server_url='test_server',
        force_convert_inputs=False,
        use_shared_memory=False,
    )

    assert stage._config is config
    assert stage._kwargs == {
        "model_name": 'test_model',
        "server_url": 'test_server',
        "force_convert_inputs": False,
        "use_shared_memory": False
    }

    # Intentionally not checking the `_requires_seg_ids` value at it appears to not be used


@pytest.mark.use_python
@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_inference_stage_get_inference_worker(config: Config, import_mod: typing.List[typing.Any]):
    inference_mod = import_mod[0]

    stage = inference_mod.LogParsingInferenceStage(
        config,
        model_name='test_model',
        server_url='test_server',
        force_convert_inputs=False,
        use_shared_memory=False,
    )

    stage._kwargs.update({'inout_mapping': {'test': 'this'}})

    worker = stage._get_inference_worker(inf_queue=ProducerConsumerQueue())
    _check_worker(inference_mod, worker)


@pytest.mark.use_python
@pytest.mark.usefixtures("manual_seed")
@pytest.mark.import_mod([
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py'),
    os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'messages.py')
])
@pytest.mark.parametrize("mess_offset,mess_count,offset,count", [(0, 5, 0, 5), (5, 5, 5, 5)])
def test_log_parsing_inference_stage_convert_one_response(config: Config,
                                                          import_mod: typing.List[typing.Any],
                                                          filter_probs_df: typing.Union[pd.DataFrame, cudf.DataFrame],
                                                          mess_offset,
                                                          mess_count,
                                                          offset,
                                                          count):
    inference_mod, messages_mod = import_mod

    ttl_count = len(filter_probs_df)

    input_res = build_response_mem(messages_mod, os.path.join(TEST_DIRS.tests_data_dir, 'log_parsing'))

    # confidences, labels & input_ids all have the same shape
    num_cols = input_res.confidences.shape[1]
    input_mem = messages_mod.PostprocMemoryLogParsing(
        count=ttl_count,
        confidences=cp.zeros((ttl_count, num_cols), dtype=cp.float32),
        input_ids=cp.zeros((ttl_count, num_cols), dtype=cp.float32),
        labels=cp.zeros((ttl_count, num_cols), dtype=cp.float32),
        seq_ids=cp.zeros((ttl_count, 3), dtype=cp.uint32),
    )

    input_inf = build_inf_message(filter_probs_df,
                                  mess_offset=mess_offset,
                                  mess_count=mess_count,
                                  offset=offset,
                                  count=count,
                                  num_cols=num_cols)

    output_msg = inference_mod.LogParsingInferenceStage._convert_one_response(input_mem, input_inf, input_res)

    assert isinstance(output_msg, messages_mod.MultiPostprocLogParsingMessage)
    assert output_msg.meta is input_inf.meta
    assert output_msg.memory is input_mem
    assert output_msg.mess_offset == mess_offset
    assert output_msg.mess_count == mess_count
    assert output_msg.offset == offset
    assert output_msg.count == count

    assert (output_msg.seq_ids == input_inf.seq_ids).all()
    assert (output_msg.confidences == input_res.confidences).all()
    assert (output_msg.labels == input_res.labels).all()

    # Ensure we didn't write to the memory outside of the [offset:offset+count] bounds
    tensors = input_mem.get_tensors()
    for (tensor_name, tensor) in tensors.items():
        assert (tensor[0:offset] == 0).all(), f"Out of bounds values for {tensor_name}"
        assert (tensor[offset + count:] == 0).all(), f"Out of bounds values for {tensor_name}"
