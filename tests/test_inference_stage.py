#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cudf

from morpheus.messages import ResponseMemoryProbs
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_response_message import MultiResponseProbsMessage
from morpheus.stages.inference.inference_stage import InferenceStage
from utils import IW


class InferenceStage(InferenceStage):
    # Subclass InferenceStage to implement the abstract methods
    def _get_inference_worker(self, pq):
        # Intentionally calling the abc empty method for coverage
        super()._get_inference_worker(pq)
        return IW(pq)


def _mk_message(mess_offset=0, mess_count=1, offset=0, count=1):
    total_message_count = mess_offset + mess_count
    total_tensor_count = offset + count

    df = cudf.DataFrame(list(range(total_message_count)), columns=["col1"])

    m = MultiInferenceMessage(meta=MessageMeta(df),
                              mess_offset=mess_offset,
                              mess_count=mess_count,
                              memory=InferenceMemory(count=total_tensor_count,
                                                     tensors={
                                                         "probs":
                                                             cp.random.rand(total_tensor_count, 2),
                                                         "seq_ids":
                                                             cp.tile(
                                                                 cp.expand_dims(cp.arange(
                                                                     mess_offset, mess_offset + total_tensor_count),
                                                                                axis=1), (1, 3))
                                                     }),
                              offset=offset,
                              count=count)

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
    message_sizes = [3, 2, 1, 7, 4]
    total_size = sum(message_sizes)

    full_input = _mk_message(mess_count=total_size, count=total_size)

    input_messages = [
        full_input.get_slice(sum(message_sizes[:i]), sum(message_sizes[:i]) + size) for i,
        size in enumerate(message_sizes)
    ]

    full_output = cp.random.rand(total_size, 3)
    output_memory = []

    for i, s in enumerate(message_sizes):
        output_memory.append(
            ResponseMemoryProbs(count=s, probs=full_output[sum(message_sizes[:i]):sum(message_sizes[:i]) + s, :]))

    resp = InferenceStage._convert_response((input_messages, output_memory))
    assert resp.meta == full_input.meta
    assert resp.mess_offset == 0
    assert resp.mess_count == total_size
    assert isinstance(resp.memory, ResponseMemoryProbs)
    assert resp.offset == 0
    assert resp.count == total_size
    assert (resp.memory.probs == full_output).all()


def test_convert_response_errors():
    # Length of input messages doesn't match length of output messages
    with pytest.raises(AssertionError):
        InferenceStage._convert_response(([1, 2, 3], [1, 2]))

    # Message offst of the second message doesn't line up offset+count of the first
    mm1 = _mk_message()
    mm2 = _mk_message(mess_offset=12)

    out_msg1 = ResponseMemoryProbs(count=1, probs=cp.random.rand(1, 3))
    out_msg2 = ResponseMemoryProbs(count=1, probs=cp.random.rand(1, 3))

    with pytest.raises(AssertionError):
        InferenceStage._convert_response(([mm1, mm2], [out_msg1, out_msg2]))

    # mess_coutn and count don't match for mm2, and mm2.count != out_msg2.count
    mm = _mk_message(mess_count=2, count=2)
    mm1 = mm.get_slice(0, 1)
    mm2 = mm.get_slice(1, 2)

    out_msg1 = ResponseMemoryProbs(count=1, probs=cp.random.rand(1, 3))
    out_msg2 = ResponseMemoryProbs(count=2, probs=cp.random.rand(2, 3))

    with pytest.raises(AssertionError):
        InferenceStage._convert_response(([mm1, mm2], [out_msg1, out_msg2]))


@pytest.mark.use_python
def test_convert_one_response():
    # Test first branch where `inf.mess_count == inf.count`
    mem = ResponseMemoryProbs(4, probs=cp.zeros((4, 3)))

    inf = _mk_message(mess_count=4, count=4)
    res = ResponseMemoryProbs(count=4, probs=cp.random.rand(4, 3))

    mpm = InferenceStage._convert_one_response(MultiResponseProbsMessage.from_message(inf, memory=mem), inf, res)
    assert mpm.meta == inf.meta
    assert mpm.mess_offset == 0
    assert mpm.mess_count == 4
    assert mpm.offset == 0
    assert mpm.count == 4
    assert cp.all(mem.get_output('probs') == res.get_output("probs"))

    # Test for the second branch
    inf = _mk_message(mess_count=3, count=3)
    inf.memory.set_tensor("seq_ids", cp.array([[0], [1], [1]]))
    inf.mess_count = 2  # Get around the consistency check
    res = ResponseMemoryProbs(count=3, probs=cp.array([[0, 0.6, 0.7], [5.6, 4.4, 9.2], [4.5, 6.7, 8.9]]))

    mem = ResponseMemoryProbs(2, probs=cp.zeros((2, 3)))
    mpm = InferenceStage._convert_one_response(MultiResponseProbsMessage.from_message(inf, memory=mem), inf, res)
    assert mem.get_output('probs').tolist() == [[0, 0.6, 0.7], [5.6, 6.7, 9.2]]


def test_convert_one_response_error():
    mem = ResponseMemoryProbs(2, probs=cp.zeros((2, 2)))
    inf = _mk_message(mess_count=2, count=2)
    res = _mk_message(mess_count=1, count=1)

    with pytest.raises(AssertionError):
        InferenceStage._convert_one_response(MultiResponseProbsMessage.from_message(inf, memory=mem), inf, res.memory)
