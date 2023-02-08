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

import cupy as cp
import numpy as np
import pytest

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.multi_inference_message import InferenceMemory
from morpheus.messages.multi_inference_message import InferenceMemoryAE
from morpheus.messages.multi_inference_message import InferenceMemoryFIL
from morpheus.messages.multi_inference_message import InferenceMemoryNLP
from morpheus.messages.multi_response_message import ResponseMemory
from morpheus.messages.multi_response_message import ResponseMemoryAE
from morpheus.messages.multi_response_message import ResponseMemoryProbs
from morpheus.messages.tensor_memory import TensorMemory
from utils import TEST_DIRS

INPUT_FILE = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')


def compare_tensors(t1, t2):
    assert sorted(t1.keys()) == sorted(t2.keys())
    for (k, v1) in t1.items():
        assert (v1 == t2[k]).all()


def check_tensor_memory(cls, count, tensors):
    other_tensors = {'ones': cp.ones(count), 'zeros': cp.zeros(count)}

    m = cls(count)
    assert m.count == count
    assert m.get_tensors() == {}

    m.set_tensors(tensors)
    compare_tensors(m.get_tensors(), tensors)

    m.set_tensors(other_tensors)
    compare_tensors(m.get_tensors(), other_tensors)

    m = cls(count, tensors)
    assert m.count == count
    compare_tensors(m.get_tensors(), tensors)

    m.set_tensors(other_tensors)
    compare_tensors(m.get_tensors(), other_tensors)


def test_tensor_memory(config):
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
def test_inference_memory_ae(config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input = cp.array(test_data[:, 0])
    seq_ids = cp.array(test_data[:, 1])
    m = InferenceMemoryAE(count, input=input, seq_ids=seq_ids)

    assert m.count == count
    compare_tensors(m.get_tensors(), {'input': input, 'seq_ids': seq_ids})
    assert (m.input == input).all()
    assert (m.seq_ids == seq_ids).all()


def test_inference_memory_fil(config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input_0 = cp.array(test_data[:, 0])
    seq_ids = cp.array(test_data[:, 1])
    m = InferenceMemoryFIL(count, input__0=input_0, seq_ids=seq_ids)

    assert m.count == count
    compare_tensors(m.get_tensors(), {'input__0': input_0, 'seq_ids': seq_ids})
    assert (m.input__0 == input_0).all()
    assert (m.seq_ids == seq_ids).all()


def test_inference_memory_nlp(config):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    input_ids = cp.array(test_data[:, 0])
    input_mask = cp.array(test_data[:, 1])
    seq_ids = cp.array(test_data[:, 2])
    m = InferenceMemoryNLP(count, input_ids=input_ids, input_mask=input_mask, seq_ids=seq_ids)

    assert m.count == count
    compare_tensors(m.get_tensors(), {'input_ids': input_ids, 'input_mask': input_mask, 'seq_ids': seq_ids})
    assert (m.input_ids == input_ids).all()
    assert (m.input_mask == input_mask).all()
    assert (m.seq_ids == seq_ids).all()


def check_response_memory_probs_and_ae(cls):
    test_data = cp.array(np.loadtxt(INPUT_FILE, delimiter=",", skiprows=1))
    count = test_data.shape[0]

    m = cls(count=count, probs=test_data)
    assert m.count == count
    compare_tensors(m.get_tensors(), {'probs': test_data})
    assert (m.probs == test_data).all()
    return m


@pytest.mark.use_python
def test_response_memory_ae(config):
    m = check_response_memory_probs_and_ae(ResponseMemoryAE)

    assert m.user_id == ""
    assert m.explain_df is None

    df = read_file_to_df(INPUT_FILE, file_type=FileTypes.Auto, df_type='pandas')
    m.user_id = "testy"
    m.explain_df = df

    assert m.user_id == "testy"
    assert (m.explain_df.values == df.values).all()


def test_response_memory_probs(config):
    check_response_memory_probs_and_ae(ResponseMemoryProbs)
