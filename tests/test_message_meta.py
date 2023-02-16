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

import operator
import os

import numpy as np
import pytest

from morpheus._lib.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.io.serializers import df_to_csv
from morpheus.messages.message_meta import MessageMeta
from utils import TEST_DIRS


def test_mutable_dataframe(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

    meta = MessageMeta(read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf'))

    with meta.mutable_dataframe() as df:
        df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] == 47


def test_using_ctx_outside_with_block(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

    meta = MessageMeta(read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf'))

    ctx = meta.mutable_dataframe()

    # ctx.fillna() & ctx.col
    pytest.raises(AttributeError, getattr, ctx, 'fillna')

    # ctx[col]
    pytest.raises(AttributeError, operator.getitem, ctx, 'col')

    # ctx[col] = 5
    pytest.raises(AttributeError, operator.setitem, ctx, 'col', 5)


def test_copy_dataframe(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

    meta = MessageMeta(read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf'))

    meta.copy_dataframe()['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47

    meta.df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47


def test_make_from_file(config, tmp_path):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs_w_id_col.csv")
    out_file = os.path.join(tmp_path, 'results.csv')
    meta = MessageMeta.make_from_file(input_file)
    with meta.mutable_dataframe() as df:
        assert list(df.columns) == ['v1', 'v2', 'v3', 'v4']

        with open(out_file, 'w') as fh:
            fh.writelines(df_to_csv(df, include_header=True, include_index_col=True))

    input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
    output_data = np.loadtxt(out_file, delimiter=",", skiprows=1)
    output_data = np.around(output_data, 2)
    assert output_data.tolist() == input_data.tolist()
