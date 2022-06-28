#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
from unittest import mock

import cupy as cp
import pytest

from morpheus.messages import ResponseMemoryProbs
from morpheus.stages.inference import inference_stage
from utils import IW


class InferenceStage(inference_stage.InferenceStage):
    # Subclass InferenceStage to implement the abstract methods
    def _get_inference_worker(self, pq):
        # Intentionally calling the abc empty method for coverage
        super()._get_inference_worker(pq)
        return IW(pq)


def _mk_message(count=1, mess_count=1, offset=0, mess_offset=0):
    m = mock.MagicMock()
    m.count = count
    m.meta.count = count
    m.offset = offset
    m.mess_offset = mess_offset
    m.mess_count = mess_count
    m.probs = cp.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])
    mask = cp.zeros(count, dtype=cp.bool_)
    mask[mess_offset:mess_offset + mess_count] = True
    m.mask = mask
    m.get_input.return_value = cp.array([[0, 1, 2], [0, 1, 2]])
    return m


def test_constructor(config):
    config.feature_length = 128
    config.num_threads = 17
    config.model_max_batch_size = 256

    inf_stage = InferenceStage(config)
    assert inf_stage._fea_length == 128
    assert inf_stage._thread_count == 17
    assert inf_stage._max_batch_size == 256
    assert inf_stage.name == "inference"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = inf_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    pytest.raises(NotImplementedError, inf_stage._get_cpp_inference_node, None)


@pytest.mark.use_python
def test_build_single(config):
    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_node
    mock_input_stream = mock.MagicMock()

    config.num_threads = 17
    inf_stage = InferenceStage(config)
    inf_stage._build_single(mock_segment, mock_input_stream)

    mock_segment.make_node_full.assert_called_once()
    mock_segment.make_edge.assert_called_once()
    assert mock_node.launch_options.pe_count == 17


@pytest.mark.use_python
def test_py_inf_fn(config):
    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_node
    mock_input_stream = mock.MagicMock()

    mock_init = mock.MagicMock()
    IW.init = mock_init

    config.num_threads = 17
    inf_stage = InferenceStage(config)
    inf_stage._build_single(mock_segment, mock_input_stream)

    py_inference_fn = mock_segment.make_node_full.call_args[0][1]

    mock_pipe = mock.MagicMock()
    mock_observable = mock.MagicMock()
    mock_observable.pipe.return_value = mock_pipe
    mock_subscriber = mock.MagicMock()
    py_inference_fn(mock_observable, mock_subscriber)

    mock_observable.pipe.assert_called_once()
    mock_pipe.subscribe.assert_called_once_with(mock_subscriber)


@pytest.mark.use_python
@mock.patch('srf.Future')
@mock.patch('morpheus.stages.inference.inference_stage.ops')
def test_py_inf_fn_on_next(mock_ops, mock_future, config):
    mock_future.return_value = mock_future
    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_node
    mock_input_stream = mock.MagicMock()

    mock_init = mock.MagicMock()
    IW.init = mock_init
    IW.process = mock.MagicMock()

    inf_stage = InferenceStage(config)
    inf_stage._build_single(mock_segment, mock_input_stream)

    py_inference_fn = mock_segment.make_node_full.call_args[0][1]

    mock_pipe = mock.MagicMock()
    mock_observable = mock.MagicMock()
    mock_observable.pipe.return_value = mock_pipe
    mock_subscriber = mock.MagicMock()
    py_inference_fn(mock_observable, mock_subscriber)

    mock_ops.map.assert_called_once()
    on_next = mock_ops.map.call_args[0][0]

    mock_message = _mk_message()

    mock_slice = _mk_message()
    mock_message.get_slice.return_value = mock_slice

    output_message = on_next(mock_message)
    assert output_message.count == 1
    assert output_message.mess_offset == 0
    assert output_message.mess_count == 1
    assert output_message.offset == 0

    mock_future.result.assert_called_once()
    mock_future.set_result.assert_not_called()

    IW.process.assert_called_once()
    set_output_fut = IW.process.call_args[0][1]
    set_output_fut(ResponseMemoryProbs(count=1, probs=cp.zeros((1, 2))))
    mock_future.set_result.assert_called_once()


@pytest.mark.use_cpp
def test_build_single_cpp(config):
    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_node
    mock_input_stream = mock.MagicMock()

    config.num_threads = 17
    inf_stage = InferenceStage(config)
    inf_stage.supports_cpp_node = lambda: True
    inf_stage._get_cpp_inference_node = lambda x: mock_node

    inf_stage._build_single(mock_segment, mock_input_stream)

    mock_segment.make_node_full.assert_not_called()
    mock_segment.make_edge.assert_called_once()
    assert mock_node.launch_options.pe_count == 17


@pytest.mark.use_cpp
def test_build_single_cpp_not_impl(config):
    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_node
    mock_input_stream = mock.MagicMock()

    inf_stage = InferenceStage(config)
    inf_stage.supports_cpp_node = lambda: True
    pytest.raises(NotImplementedError, inf_stage._build_single, mock_segment, mock_input_stream)


def test_start(config):
    inf_stage = InferenceStage(config)

    pytest.raises(AssertionError, inf_stage.start)

    inf_stage._is_built = True
    inf_stage.start()


def test_stop(config):
    mock_workers = [mock.MagicMock() for _ in range(5)]
    inf_stage = InferenceStage(config)
    inf_stage._workers = mock_workers

    inf_stage.stop()
    for w in mock_workers:
        w.stop.assert_called_once()

    assert inf_stage._inf_queue.is_closed()


def test_join(config):
    mock_workers = [mock.AsyncMock() for _ in range(5)]
    inf_stage = InferenceStage(config)
    inf_stage._workers = mock_workers

    asyncio.run(inf_stage.join())
    for w in mock_workers:
        w.join.assert_awaited_once()


def test_split_batches():
    seq_ids = cp.zeros((10, 1))
    seq_ids[2][0] = 15
    seq_ids[6][0] = 16

    mock_message = mock.MagicMock()
    mock_message.get_input.return_value = seq_ids

    out_resp = InferenceStage._split_batches(mock_message, 5)
    assert len(out_resp) == 3

    assert mock_message.get_slice.call_count == 3
    mock_message.get_slice.assert_has_calls([mock.call(0, 3), mock.call(3, 7), mock.call(7, 10)])


@pytest.mark.use_python
def test_convert_response(config):
    mm1 = _mk_message()
    mm2 = _mk_message(mess_offset=1)

    out_msg1 = _mk_message()
    out_msg1.probs = cp.array([[0.1, 0.5, 0.8]])

    out_msg2 = _mk_message(mess_offset=1)
    out_msg2.probs = cp.array([[0.1, 0.5, 0.8]])

    resp = inference_stage.InferenceStage._convert_response(([mm1, mm2], [out_msg1, out_msg2]))
    assert resp.meta == mm1.meta
    assert resp.mess_offset == 0
    assert resp.mess_count == 2
    assert isinstance(resp.memory, ResponseMemoryProbs)
    assert resp.offset == 0
    assert resp.count == 2
    assert resp.memory.probs.tolist() == [[0.1, 0.5, 0.8], [0, 0, 0]]

    mm2.count = 2
    out_msg2.probs = cp.array([[0.1, 0.5, 0.8], [4.5, 6.7, 8.9]])
    mm2.seq_ids = cp.array([[0], [1]])
    out_msg2.count = 2
    resp = inference_stage.InferenceStage._convert_response(([mm1, mm2], [out_msg1, out_msg2]))
    assert resp.meta == mm1.meta
    assert resp.mess_offset == 0
    assert resp.mess_count == 2
    assert isinstance(resp.memory, ResponseMemoryProbs)
    assert resp.offset == 0
    assert resp.count == 2
    assert resp.memory.probs.tolist() == [[0.1, 0.5, 0.8], [4.5, 6.7, 8.9]]


def test_convert_response_errors():
    # Length of input messages doesn't match length of output messages
    pytest.raises(AssertionError, inference_stage.InferenceStage._convert_response, ([1, 2, 3], [1, 2]))

    # Message offst of the second message doesn't line up offset+count of the first
    mm1 = _mk_message()
    mm2 = _mk_message(mess_offset=12)

    out_msg1 = _mk_message()
    out_msg1.probs = cp.array([[0.1, 0.5, 0.8]])

    out_msg2 = _mk_message(mess_offset=1)
    out_msg2.probs = cp.array([[0.1, 0.5, 0.8]])

    pytest.raises(AssertionError, inference_stage.InferenceStage._convert_response, ([mm1, mm2], [out_msg1, out_msg2]))

    # mess_coutn and count don't match for mm2, and mm2.count != out_msg2.count
    mm2.mess_offset = 1
    mm2.count = 2

    pytest.raises(AssertionError, inference_stage.InferenceStage._convert_response, ([mm1, mm2], [out_msg1, out_msg2]))

    # saved_count != total_mess_count
    # Unlike the other asserts that can be triggered due to bad input data
    # This one can only be triggers by a bug inside the method
    mm2 = _mk_message()
    mm2.count = mock.MagicMock()
    mm2.mess_count = mock.MagicMock()
    mm2.count.side_effect = [2, 1]
    mm2.mess_count.side_effect = [2, 1, 1]

    pytest.raises(AssertionError, inference_stage.InferenceStage._convert_response, ([mm1, mm2], [out_msg1, out_msg2]))


@pytest.mark.use_python
def test_convert_one_response(config):
    # Test first branch where `inf.mess_count == inf.count`
    mem = ResponseMemoryProbs(1, probs=cp.zeros((1, 3)))

    inf = _mk_message()
    res = ResponseMemoryProbs(count=1, probs=cp.array([[1, 2, 3]]))

    mpm = inference_stage.InferenceStage._convert_one_response(mem, inf, res)
    assert mpm.meta == inf.meta
    assert mpm.mess_offset == 0
    assert mpm.mess_count == 1
    assert mpm.memory == mem
    assert mpm.offset == 0
    assert mpm.count == 1
    assert mem.get_output('probs').tolist() == [[1.0, 2.0, 3.0]]

    # Test for the second branch
    inf.mess_count = 2
    inf.seq_ids = cp.array([[0], [1]])
    res = ResponseMemoryProbs(count=1, probs=cp.array([[0, 0.6, 0.7], [5.6, 4.4, 9.2]]))

    mem = ResponseMemoryProbs(1, probs=cp.array([[0.1, 0.5, 0.8], [4.5, 6.7, 8.9]]))
    mpm = inference_stage.InferenceStage._convert_one_response(mem, inf, res)
    assert mem.get_output('probs').tolist() == [[0.1, 0.6, 0.8], [5.6, 6.7, 9.2]]


def test_convert_one_response_error():
    mem = ResponseMemoryProbs(1, probs=cp.zeros((1, 3)))
    inf = _mk_message(mess_count=2)
    res = _mk_message(count=2)

    pytest.raises(AssertionError, inference_stage.InferenceStage._convert_one_response, mem, inf, res)
