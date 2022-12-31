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

import pytest

from morpheus._lib.file_types import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages.message_meta import MessageMeta
from utils import TEST_DIRS


def test_mutable_dataframe(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

    meta = MessageMeta(read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf'))

    with meta.mutable_dataframe() as ctx:
        ctx.df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] == 47

    pytest.raises(RuntimeError, getattr, ctx, 'df')

    copied_df = meta.copy_dataframe()
    with meta.mutable_dataframe() as ctx:
        pytest.raises(AttributeError, setattr, ctx, 'df', copied_df)


def test_copy_dataframe(config):
    input_file = os.path.join(TEST_DIRS.tests_data_dir, 'filter_probs.csv')

    meta = MessageMeta(read_file_to_df(input_file, file_type=FileTypes.Auto, df_type='cudf'))

    meta.copy_dataframe()['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47

    meta.df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47
