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

import os
import string
import typing

import cupy as cp
import numpy as np
import pytest

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.messages.memory.inference_memory import InferenceMemory
from morpheus.messages.memory.inference_memory import InferenceMemoryAE
from morpheus.messages.memory.inference_memory import InferenceMemoryFIL
from morpheus.messages.memory.inference_memory import InferenceMemoryNLP
from morpheus.messages.memory.response_memory import ResponseMemory
from morpheus.messages.memory.response_memory import ResponseMemoryAE
from morpheus.messages.memory.response_memory import ResponseMemoryProbs
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.utils.type_aliases import DataFrameType

INPUT_FILE = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

# Many of our tests require the config fixture, but don't use the value.
# pylint: disable=unused-argument


def compare_tensors(tensors1: typing.Dict[str, cp.ndarray], tensors2: typing.Dict[str, cp.ndarray]):
    assert sorted(tensors1.keys()) == sorted(tensors2.keys())
    for (k, val1) in tensors1.items():
        assert (val1 == tensors2[k]).all()


def check_tensor_memory(cls: type, count: int, tensors: typing.Dict[str, cp.ndarray]):
    other_tensors = {'ones': cp.ones(count), 'zeros': cp.zeros(count)}

    mem = cls(count=count)
    assert mem.count == count
    assert mem.get_tensors() == {}

    mem.set_tensors(tensors)
    compare_tensors(mem.get_tensors(), tensors)

    mem.set_tensors(other_tensors)
    compare_tensors(mem.get_tensors(), other_tensors)

    mem = cls(count=count, tensors=tensors)
    assert mem.count == count
    compare_tensors(mem.get_tensors(), tensors)

    mem.set_tensors(other_tensors)
    compare_tensors(mem.get_tensors(), other_tensors)

    with pytest.raises(TypeError):
        cls(count)

    with pytest.raises(TypeError):
        cls(count, tensors)


def test_tensor_memory(config: Config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    # TensorMemory expects a dictionary of {<tensor_name> : <cupy array> }
    # Convert each column into a 1d cupy array
    tensors = {}
    for col in range(test_data.shape[1]):
        tensors[string.ascii_lowercase[col]] = cp.array(test_data[:, col])

    for cls in (TensorMemory, InferenceMemory, ResponseMemory):
        check_tensor_memory(cls, count, tensors)


@pytest.mark.use_python
def test_inference_memory_ae(config: Config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input_tensor = cp.array(test_data[:, 0])
    seq_ids = cp.array(test_data[:, 1])
    mem = InferenceMemoryAE(count=count, input=input_tensor, seq_ids=seq_ids)

    assert mem.count == count
    compare_tensors(mem.get_tensors(), {'input': input_tensor, 'seq_ids': seq_ids})
    assert (mem.input == input_tensor).all()
    assert (mem.seq_ids == seq_ids).all()

    with pytest.raises(TypeError):
        InferenceMemoryAE(count, input_tensor, seq_ids)  # pylint: disable=too-many-function-args,missing-kwoa


def test_inference_memory_fil(config: Config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input_0 = cp.array(test_data[:, 0])
    seq_ids = cp.array(test_data[:, 1])
    mem = InferenceMemoryFIL(count=count, input__0=input_0, seq_ids=seq_ids)

    assert mem.count == count
    compare_tensors(mem.get_tensors(), {'input__0': input_0, 'seq_ids': seq_ids})
    assert (mem.input__0 == input_0).all()
    assert (mem.seq_ids == seq_ids).all()

    with pytest.raises(TypeError):
        InferenceMemoryFIL(count, input_0, seq_ids)  # pylint: disable=too-many-function-args,missing-kwoa


def test_inference_memory_nlp(config: Config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input_ids = cp.array(test_data[:, 0])
    input_mask = cp.array(test_data[:, 1])
    seq_ids = cp.array(test_data[:, 2])
    mem = InferenceMemoryNLP(count=count, input_ids=input_ids, input_mask=input_mask, seq_ids=seq_ids)

    assert mem.count == count
    compare_tensors(mem.get_tensors(), {'input_ids': input_ids, 'input_mask': input_mask, 'seq_ids': seq_ids})
    assert (mem.input_ids == input_ids).all()
    assert (mem.input_mask == input_mask).all()
    assert (mem.seq_ids == seq_ids).all()

    with pytest.raises(TypeError):
        InferenceMemoryNLP(count, input_ids, input_mask, seq_ids)  # pylint: disable=too-many-function-args,missing-kwoa


def check_response_memory_probs_and_ae(cls: type):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    mem = cls(count=count, probs=test_data)
    assert mem.count == count
    compare_tensors(mem.get_tensors(), {'probs': test_data})
    assert (mem.get_output('probs') == test_data).all()

    with pytest.raises(TypeError):
        cls(count, test_data)

    return mem


@pytest.mark.use_python
def test_response_memory_ae(config: Config, filter_probs_df: DataFrameType):
    mem = check_response_memory_probs_and_ae(ResponseMemoryAE)

    assert mem.user_id == ""
    assert mem.explain_df is None

    mem.user_id = "testy"
    mem.explain_df = filter_probs_df

    assert mem.user_id == "testy"
    assert (mem.explain_df.values == filter_probs_df.values).all()


def test_response_memory_probs(config: Config):
    check_response_memory_probs_and_ae(ResponseMemoryProbs)


@pytest.mark.parametrize("tensor_cls", [TensorMemory, InferenceMemory, ResponseMemory])
def test_constructor_length_error(config: Config, tensor_cls: type):
    count = 10
    tensors = {"a": cp.zeros(count), "b": cp.ones(count)}

    with pytest.raises(ValueError):
        tensor_cls(count=count - 1, tensors=tensors)


@pytest.mark.parametrize("tensor_cls", [TensorMemory, InferenceMemory, ResponseMemory])
def test_set_tensor_length_error(config: Config, tensor_cls: type):
    count = 10
    mem = tensor_cls(count=count)

    with pytest.raises(ValueError):
        mem.set_tensor('a', cp.zeros(count + 1))


@pytest.mark.parametrize("tensor_cls", [TensorMemory, InferenceMemory, ResponseMemory])
def test_set_tensors_length_error(config: Config, tensor_cls: type):
    count = 10
    tensors = {"a": cp.zeros(count), "b": cp.ones(count)}
    mem = tensor_cls(count=count + 1)

    with pytest.raises(ValueError):
        mem.set_tensors(tensors)


@pytest.mark.parametrize("tensor_cls", [TensorMemory, InferenceMemory, ResponseMemory])
@pytest.mark.parametrize(
    "shape",
    [
        (536870912, 1),  # bytesize > 2**31
        (134217728, 4)  # bytesize > 2**31 and element count > 2**31
    ])
def test_tensorindex_bug(config: Config, tensor_cls: type, shape: typing.Tuple[int, int]):
    """
    Test for issue #1004. We use a 32bit signed integer for shape and strides, but we shouldn't for element counts and
    byte sizes.
    """
    tensors = {"a": cp.zeros(shape, dtype=np.float32)}

    mem = tensor_cls(count=shape[0], tensors=tensors)
    tensor_a = mem.get_tensor('a')
    assert tensor_a.shape == shape
    assert tensor_a.nbytes == shape[0] * shape[1] * 4
