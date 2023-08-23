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

from _utils.inference_worker import IW
from morpheus.messages import ResponseMemory
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_inference_message import MultiInferenceMessage
from morpheus.messages.multi_response_message import MultiResponseMessage
from morpheus.stages.inference.inference_stage import InferenceStage


class InferenceStageT(InferenceStage):
    # Subclass InferenceStage to implement the abstract methods
    def _get_inference_worker(self, inf_queue):
        # Intentionally calling the abc empty method for coverage
        super()._get_inference_worker(inf_queue)
        return IW(inf_queue)


def _mk_message(mess_offset=0, mess_count=1, offset=0, count=1):
    total_message_count = mess_offset + mess_count
    total_tensor_count = offset + count

    df = cudf.DataFrame(list(range(total_message_count)), columns=["col1"])

    msg = MultiInferenceMessage(meta=MessageMeta(df),
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

    return msg


def test_constructor(config):
    config.feature_length = 128
    config.num_threads = 17
    config.model_max_batch_size = 256

    inf_stage = InferenceStageT(config)
    assert inf_stage._fea_length == 128
    assert inf_stage._thread_count == 17
    assert inf_stage._max_batch_size == 256
    assert inf_stage.name == "inference"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = inf_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    pytest.raises(NotImplementedError, inf_stage._get_cpp_inference_node, None)


def test_stop(config):
    mock_workers = [mock.MagicMock() for _ in range(5)]
    inf_stage = InferenceStageT(config)
    inf_stage._workers = mock_workers  # pylint: disable=attribute-defined-outside-init

    inf_stage.stop()
    for worker in mock_workers:
        worker.stop.assert_called_once()

    assert inf_stage._inf_queue.is_closed()


def test_join(config):
    mock_workers = [mock.AsyncMock() for _ in range(5)]
    inf_stage = InferenceStageT(config)
    inf_stage._workers = mock_workers  # pylint: disable=attribute-defined-outside-init

    asyncio.run(inf_stage.join())
    for worker in mock_workers:
        worker.join.assert_awaited_once()


def test_split_batches():
    seq_ids = cp.zeros((10, 1))
    seq_ids[2][0] = 15
    seq_ids[6][0] = 16

    mock_message = mock.MagicMock()
    mock_message.get_input.return_value = seq_ids

    out_resp = InferenceStageT._split_batches(mock_message, 5)
    assert len(out_resp) == 3

    assert mock_message.get_slice.call_count == 3
    mock_message.get_slice.assert_has_calls([mock.call(0, 3), mock.call(3, 7), mock.call(7, 10)])


@pytest.mark.use_python
def test_convert_response():
    # Pylint currently fails to work with classmethod: https://github.com/pylint-dev/pylint/issues/981
    # pylint: disable=no-member

    message_sizes = [3, 2, 1, 7, 4]
    total_size = sum(message_sizes)

    full_input = _mk_message(mess_count=total_size, count=total_size)

    input_messages = [
        full_input.get_slice(sum(message_sizes[:i]), sum(message_sizes[:i]) + size) for i,
        size in enumerate(message_sizes)
    ]

    full_output = cp.random.rand(total_size, 3)
    output_memory = []

    for i, count in enumerate(message_sizes):
        output_memory.append(
            ResponseMemory(count=count,
                           tensors={"probs": full_output[sum(message_sizes[:i]):sum(message_sizes[:i]) + count, :]}))

    resp = InferenceStageT._convert_response((input_messages, output_memory))
    assert isinstance(resp, MultiResponseMessage)
    assert resp.meta == full_input.meta
    assert resp.mess_offset == 0
    assert resp.mess_count == total_size
    assert isinstance(resp.memory, TensorMemory)
    assert resp.offset == 0
    assert resp.count == total_size
    assert (resp.memory.get_tensor("probs") == full_output).all()


def test_convert_response_errors():
    # Length of input messages doesn't match length of output messages
    with pytest.raises(AssertionError):
        InferenceStageT._convert_response(([1, 2, 3], [1, 2]))

    # Message offst of the second message doesn't line up offset+count of the first
    msg1 = _mk_message()
    msg2 = _mk_message(mess_offset=12)

    out_msg1 = ResponseMemory(count=1, tensors={"probs": cp.random.rand(1, 3)})
    out_msg2 = ResponseMemory(count=1, tensors={"probs": cp.random.rand(1, 3)})

    with pytest.raises(AssertionError):
        InferenceStageT._convert_response(([msg1, msg2], [out_msg1, out_msg2]))

    # mess_coutn and count don't match for msg2, and msg2.count != out_msg2.count
    msg = _mk_message(mess_count=2, count=2)
    msg1 = msg.get_slice(0, 1)
    msg2 = msg.get_slice(1, 2)

    out_msg1 = ResponseMemory(count=1, tensors={"probs": cp.random.rand(1, 3)})
    out_msg2 = ResponseMemory(count=2, tensors={"probs": cp.random.rand(2, 3)})

    with pytest.raises(AssertionError):
        InferenceStageT._convert_response(([msg1, msg2], [out_msg1, out_msg2]))


@pytest.mark.use_python
def test_convert_one_response():
    # Pylint currently fails to work with classmethod: https://github.com/pylint-dev/pylint/issues/981
    # pylint: disable=no-member

    # Test first branch where `inf.mess_count == inf.count`
    mem = ResponseMemory(count=4, tensors={"probs": cp.zeros((4, 3))})

    inf = _mk_message(mess_count=4, count=4)
    res = ResponseMemory(count=4, tensors={"probs": cp.random.rand(4, 3)})

    mpm = InferenceStageT._convert_one_response(MultiResponseMessage.from_message(inf, memory=mem), inf, res)
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
    res = ResponseMemory(count=3, tensors={"probs": cp.array([[0, 0.6, 0.7], [5.6, 4.4, 9.2], [4.5, 6.7, 8.9]])})

    mem = ResponseMemory(count=2, tensors={"probs": cp.zeros((2, 3))})
    mpm = InferenceStageT._convert_one_response(MultiResponseMessage.from_message(inf, memory=mem), inf, res)
    assert mem.get_output('probs').tolist() == [[0, 0.6, 0.7], [5.6, 6.7, 9.2]]


def test_convert_one_response_error():
    mem = ResponseMemory(count=2, tensors={"probs": cp.zeros((2, 2))})
    inf = _mk_message(mess_count=2, count=2)
    res = _mk_message(mess_count=1, count=1)

    with pytest.raises(AssertionError):
        InferenceStageT._convert_one_response(MultiResponseMessage.from_message(inf, memory=mem), inf, res.memory)
