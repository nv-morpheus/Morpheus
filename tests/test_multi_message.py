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

import os

import cupy
import pytest

from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage
from utils import TEST_DIRS

# TODO: These should work with the CPP impls as well


@pytest.mark.use_python
def test_masking(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')
    mask = cupy.zeros(len(df), dtype=cupy.bool_)
    mask[2:6] = True

    meta = MessageMeta(df)
    assert meta.count == len(df)

    mm = MultiMessage(meta, 0, len(df))
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == len(df)

    assert len(mm.mask) == len(df)
    mm.mask = mask
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == 4

    # Add 3 more rows to our mask
    mm.mask[12:15] = True
    assert len(mm.get_meta()) == 7


@pytest.mark.use_python
def test_start_stop(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')

    meta = MessageMeta(df)
    assert meta.count == len(df)

    mm = MultiMessage(meta, 2, 4)
    assert mm.meta.count == len(df)
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == 4

    expected_mask = []
    for i in range(len(df)):
        expected_mask.append(i >= 2 and i < 6)

    assert mm.mask.tolist() == expected_mask


@pytest.mark.use_python
def test_set_meta(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')
    mask = cupy.zeros(len(df), dtype=cupy.bool_)
    mask[2:6] = True
    mask[12:15] = True

    meta = MessageMeta(df)
    mm = MultiMessage(meta, 0, len(df))
    mm.mask = mask
    assert len(mm.get_meta()) == 7

    values = list(range(7))
    mm.set_meta('v2', values)

    assert mm.get_meta_list('v2') == values

    rev_values = list(reversed(values))
    mm.set_meta('new_col', rev_values)
    assert mm.get_meta_list('new_col') == rev_values
