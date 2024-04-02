# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import types
import typing
from unittest import mock

import cupy as cp
import numpy as np
import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.messages import InferenceMemoryNLP
from morpheus.messages import MessageMeta
from morpheus.messages import MultiInferenceNLPMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages import TensorMemory
from morpheus.stages.inference.triton_inference_stage import TritonInferenceWorker
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue
from morpheus.utils.type_aliases import DataFrameType


def build_response_mem(log_test_data_dir: str) -> TensorMemory:
    # we have tensor data for the first five rows
    count = 5
    tensors = {}
    for tensor_name in ['confidences', 'labels']:
        tensor_file = os.path.join(log_test_data_dir, f'{tensor_name}.csv')
        host_data = np.loadtxt(tensor_file, delimiter=',')
        tensors[tensor_name] = cp.asarray(host_data)

    return TensorMemory(count=count, tensors=tensors)


def build_resp_message(df: DataFrameType, num_cols: int = 2) -> MultiResponseMessage:
    count = len(df)
    seq_ids = cp.zeros((count, 3), dtype=cp.uint32)
    seq_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
    seq_ids[:, 2] = 42

    meta = MessageMeta(df)
    mem = TensorMemory(count=count,
                       tensors={
                           'confidences': cp.zeros((count, num_cols)),
                           'labels': cp.zeros((count, num_cols)),
                           'input_ids': cp.zeros((count, num_cols), dtype=cp.float32),
                           'seq_ids': seq_ids
                       })

    return MultiResponseMessage(meta=meta, mess_offset=0, mess_count=count, memory=mem, offset=0, count=count)


def build_inf_message(df: DataFrameType,
                      mess_offset: int,
                      mess_count: int,
                      offset: int,
                      count: int,
                      num_cols: int = 2) -> MultiInferenceNLPMessage:
    assert count >= mess_count
    tensor_length = offset + count
    seq_ids = cp.zeros((tensor_length, 3), dtype=cp.uint32)

    id_range = cp.arange(mess_offset, mess_offset + mess_count, dtype=cp.uint32)
    seq_ids[offset:offset + mess_count, 0] = id_range
    if (count != mess_count):  # Repeat the last id
        seq_ids[offset + mess_count:offset + count, 0] = id_range[-1]

    seq_ids[:, 2] = 42

    meta = MessageMeta(df)
    mem = InferenceMemoryNLP(count=tensor_length,
                             input_ids=cp.zeros((tensor_length, num_cols), dtype=cp.float32),
                             input_mask=cp.zeros((tensor_length, num_cols), dtype=cp.float32),
                             seq_ids=seq_ids)

    return MultiInferenceNLPMessage(meta=meta,
                                    mess_offset=mess_offset,
                                    mess_count=mess_count,
                                    memory=mem,
                                    offset=offset,
                                    count=count)


def _check_worker(inference_mod: types.ModuleType, worker: TritonInferenceWorker):
    assert isinstance(worker, TritonInferenceWorker)
    assert isinstance(worker, inference_mod.TritonInferenceLogParsing)
    assert worker._model_name == 'test_model'
    assert worker._server_url == 'test_server'
    assert not worker._force_convert_inputs
    assert not worker._use_shared_memory
    assert worker.needs_logits


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_triton_inference_log_parsing_constructor(config: Config,
                                                              import_mod: typing.List[types.ModuleType]):
    inference_mod = import_mod[0]
    worker = inference_mod.TritonInferenceLogParsing(inf_queue=ProducerConsumerQueue(),
                                                     c=config,
                                                     model_name='test_model',
                                                     server_url='test_server',
                                                     force_convert_inputs=False,
                                                     use_shared_memory=False,
                                                     input_mapping={'test': 'this'},
                                                     needs_logits=True)

    _check_worker(inference_mod, worker)


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
@pytest.mark.parametrize("mess_offset,mess_count,offset,count", [(0, 20, 0, 20), (5, 10, 5, 10)])
def test_log_parsing_triton_inference_log_parsing_build_output_message(config: Config,
                                                                       filter_probs_df: DataFrameType,
                                                                       import_mod: typing.List[types.ModuleType],
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
    assert msg.get_tensor('confidences').shape == (count, 2)
    assert msg.get_tensor('labels').shape == (count, 2)
    assert msg.get_tensor('input_ids').shape == (count, 2)
    assert msg.get_tensor('seq_ids').shape == (count, 3)


@pytest.mark.import_mod([os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py')])
def test_log_parsing_inference_stage_get_inference_worker(config: Config, import_mod: typing.List[types.ModuleType]):
    inference_mod = import_mod[0]

    stage = inference_mod.LogParsingInferenceStage(config,
                                                   model_name='test_model',
                                                   server_url='test_server',
                                                   force_convert_inputs=False,
                                                   use_shared_memory=False,
                                                   input_mapping={'test': 'this'})

    worker = stage._get_inference_worker(inf_queue=ProducerConsumerQueue())
    _check_worker(inference_mod, worker)


@pytest.mark.use_cudf
@pytest.mark.usefixtures("manual_seed", "config")
@pytest.mark.import_mod(os.path.join(TEST_DIRS.examples_dir, 'log_parsing', 'inference.py'))
@pytest.mark.parametrize("mess_offset,mess_count,offset,count", [(0, 5, 0, 5), (5, 5, 0, 5)])
def test_log_parsing_inference_stage_convert_one_response(import_mod: typing.List[types.ModuleType],
                                                          filter_probs_df: DataFrameType,
                                                          mess_offset,
                                                          mess_count,
                                                          offset,
                                                          count):
    inference_mod = import_mod

    input_res = build_response_mem(os.path.join(TEST_DIRS.tests_data_dir, 'examples/log_parsing'))

    # confidences, labels & input_ids all have the same shape
    num_cols = input_res.get_tensor('confidences').shape[1]
    resp_msg = build_resp_message(filter_probs_df, num_cols=num_cols)

    orig_tensors = {k: v.copy() for (k, v) in resp_msg.memory.get_tensors().items()}

    input_inf = build_inf_message(filter_probs_df,
                                  mess_offset=mess_offset,
                                  mess_count=mess_count,
                                  offset=offset,
                                  count=count,
                                  num_cols=num_cols)

    output_msg = inference_mod.LogParsingInferenceStage._convert_one_response(resp_msg, input_inf, input_res)

    assert isinstance(output_msg, MultiResponseMessage)
    assert output_msg.meta is input_inf.meta
    assert output_msg.mess_offset == mess_offset
    assert output_msg.mess_count == mess_count
    assert output_msg.offset == offset
    assert output_msg.count == count

    assert (output_msg.get_tensor('seq_ids') == input_inf.get_tensor('seq_ids')).all()
    assert (output_msg.get_tensor('input_ids') == input_inf.get_tensor('input_ids')).all()
    assert (output_msg.get_tensor('confidences') == input_res.get_tensor('confidences')).all()
    assert (output_msg.get_tensor('labels') == input_res.get_tensor('labels')).all()

    # Ensure we didn't write to the memory outside of the [offset:offset+count] bounds
    tensors = resp_msg.memory.get_tensors()
    for (tensor_name, tensor) in tensors.items():
        orig_tensor = orig_tensors[tensor_name]

        error_msg = f"Out of bounds values for {tensor_name}"
        assert (tensor[0:offset] == orig_tensor[0:offset]).all(), error_msg
        assert (tensor[offset + count:] == orig_tensor[offset + count:]).all(), error_msg
