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

import re

import pytest

import cudf

from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.output.serialize import SerializeStage


@pytest.mark.use_python
def test_fixed_columns(config):
    df1 = cudf.DataFrame()
    df1['apples'] = range(0, 4)
    df1['pears'] = range(5, 9)
    df1['apple_sauce'] = range(4, 0, -1)
    mm1 = MultiMessage(MessageMeta(df1), 0, 1)

    df2 = cudf.DataFrame()
    df2['apples'] = range(4, 7)
    df2['applause'] = range(9, 6, -1)
    df2['pears'] = range(7, 10)
    df2['apple_sauce'] = range(6, 3, -1)
    mm2 = MultiMessage(MessageMeta(df2), 0, 1)

    include_re_str = '^app.*'
    include_re = re.compile(include_re_str)
    s = SerializeStage(config, include=[include_re_str], fixed_columns=True)
    meta1 = s.convert_to_df(mm1, include_columns=include_re, exclude_columns=[])
    meta2 = s.convert_to_df(mm2, include_columns=include_re, exclude_columns=[])

    assert meta1.df.columns.to_list() == ['apples', 'apple_sauce']
    assert meta2.df.columns.to_list() == ['apples', 'apple_sauce']

    s = SerializeStage(config, include=[include_re_str], fixed_columns=False)
    meta1 = s.convert_to_df(mm1, include_columns=include_re, exclude_columns=[])
    meta2 = s.convert_to_df(mm2, include_columns=include_re, exclude_columns=[])

    assert meta1.df.columns.to_list() == ['apples', 'apple_sauce']
    assert meta2.df.columns.to_list() == ['apples', 'applause', 'apple_sauce']
