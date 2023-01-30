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
from morpheus.messages import InferenceMemory
from morpheus.messages import ResponseMemory
from morpheus.messages.tensor_memory import TensorMemory
from utils import TEST_DIRS


def check_tensor_memory(cls, count, tensors):
    other_tensors = {'ones': cp.ones(count), 'zeros': cp.zeros(count)}

    m = cls(count)
    assert m.count == count
    assert m.tensors == {}

    m.tensors = tensors
    assert m.tensors == tensors

    m.tensors = other_tensors
    assert m.tensors == other_tensors

    m = cls(count, tensors)
    assert m.count == count
    assert m.tensors == tensors

    m.tensors = other_tensors
    assert m.tensors == other_tensors


def test_copy_ranges(config, df_type):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    test_data = cp.array(np.loadtxt(input_file, delimiter=",", skiprows=1))

    # TensorMemory expects a dictionary of {<tensor_name> : <cupy array> }
    # Convert each column into a 1d cupy array
    tensors = {}
    for col in range(test_data.shape[1]):
        tensors[string.ascii_lowercase[col]] = test_data[:, col]

    count = test_data.shape[0]

    for cls in (TensorMemory, InferenceMemory, ResponseMemory):
        check_tensor_memory(cls, count, tensors)
