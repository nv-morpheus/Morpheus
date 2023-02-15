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

import pytest

from morpheus._lib.common import FileTypes
from morpheus.config import CppConfig
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import df_to_csv
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage
from utils import TEST_DIRS
from utils import create_df_with_dup_ids


@pytest.mark.parametrize('df_type', ['cudf', 'pandas'])
def test_copy_ranges(config, df_type):
    if CppConfig.get_should_use_cpp() and df_type == 'pandas':
        pytest.skip("Pandas dataframes not supported in C++ mode")

    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type=df_type)

    meta = MessageMeta(df)
    assert meta.has_unique_index()
    assert meta.count == len(df)

    mm = MultiMessage(meta, 0, meta.count)
    assert mm.meta.count == len(df)
    assert len(mm.get_meta()) == meta.count

    mm2 = mm.copy_ranges([(2, 6)])
    assert len(mm2.meta.df) == 4
    assert mm2.meta.count == 4
    assert len(mm2.get_meta()) == 4
    assert mm2.meta is not meta
    assert mm2.meta.df is not df

    # slice two different ranges of rows
    mm3 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm3.meta.df) == 7
    assert mm3.meta.count == 7
    assert len(mm3.get_meta()) == 7
    assert mm3.meta is not meta
    assert mm3.meta is not mm2.meta
    assert mm3.meta.df is not df
    assert mm3.meta.df is not mm2.meta.df


def test_set_meta(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')
    df = read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf')

    meta = MessageMeta(df)
    assert meta.has_unique_index()
    mm = MultiMessage(meta, 0, meta.count)

    mm2 = mm.copy_ranges([(2, 6), (12, 15)])
    assert len(mm2.get_meta()) == 7

    values = list(range(7))
    mm2.set_meta('v2', values)

    assert mm2.get_meta_list('v2') == values

    assert mm2.get_meta_list(None) == mm2.get_meta().to_arrow().to_pylist()


@pytest.mark.parametrize("dup_row", [0, 1, 8, 18, 19])  # test for dups at the front, middle and the tail
def test_duplicate_ids(config, tmp_path, dup_row):
    """
    Test for dataframe with duplicate IDs issue #686
    """
    dup_file = create_df_with_dup_ids(tmp_path, dup_row=dup_row)

    dup_df = read_file_to_df(dup_file, file_type=FileTypes.Auto, df_type='cudf')
    assert not dup_df.index.is_unique

    meta = MessageMeta(dup_df)
    assert not meta.has_unique_index()

    assert meta.count == len(dup_df)

    with meta.mutable_dataframe() as mut_df:
        assert len(mut_df) == len(dup_df)

    # C++ fails here the copy returns 22 rows
    assert len(meta.copy_dataframe()) == len(dup_df)

    mm = MultiMessage(meta, 0, meta.count)

    # Python fails here mm.get_meta_list(None) returns 22 rows
    assert mm.get_meta_list(None) == dup_df.to_arrow().to_pylist()

    meta = MessageMeta(dup_df)
    mm = MultiMessage(meta, 0, len(meta.df))

    # Fails on an assert because len(meta.df) returned 22
    assert mm.get_meta_list(None) == dup_df.to_arrow().to_pylist()
