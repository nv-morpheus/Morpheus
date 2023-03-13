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

import pytest

from morpheus.messages.message_meta import MessageMeta
from utils import TEST_DIRS


def test_mutable_dataframe(config, filter_probs_df):
    meta = MessageMeta(filter_probs_df)

    with meta.mutable_dataframe() as df:
        df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] == 47


def test_using_ctx_outside_with_block(config, filter_probs_df):
    meta = MessageMeta(filter_probs_df)

    ctx = meta.mutable_dataframe()

    # ctx.fillna() & ctx.col
    pytest.raises(AttributeError, getattr, ctx, 'fillna')

    # ctx[col]
    pytest.raises(AttributeError, operator.getitem, ctx, 'col')

    # ctx[col] = 5
    pytest.raises(AttributeError, operator.setitem, ctx, 'col', 5)


def test_copy_dataframe(config, filter_probs_df):
    meta = MessageMeta(filter_probs_df)

    meta.copy_dataframe()['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47

    meta.df['v2'][3] = 47

    assert meta.copy_dataframe()['v2'][3] != 47
    assert meta.df != 47
