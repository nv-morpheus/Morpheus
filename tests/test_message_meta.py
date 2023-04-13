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
import typing

import pandas as pd
import pytest

import cudf

from dataset_manager import DatasetManager
from morpheus.messages.message_meta import MessageMeta


@pytest.fixture(scope="function", params=["normal", "skip", "dup", "down", "updown"])
def index_type(request: pytest.FixtureRequest) -> typing.Literal["normal", "skip", "dup", "down", "updown"]:
    return request.param


@pytest.fixture(scope="function")
def df(
    dataset: DatasetManager, index_type: typing.Literal['normal', 'skip', 'dup', 'down',
                                                        'updown']) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
    filter_probs_df = dataset["filter_probs.csv"]
    if (index_type == "normal"):
        return filter_probs_df
    elif (index_type == "skip"):
        # Skip some rows
        return filter_probs_df.iloc[::3, :].copy()
    elif (index_type == "dup"):
        # Duplicate
        return dataset.dup_index(filter_probs_df, count=2)
    elif (index_type == "down"):
        # Reverse
        return filter_probs_df.iloc[::-1, :].copy()
    elif (index_type == "updown"):
        # Go up then down
        down = filter_probs_df.iloc[::-1, :].copy()

        # Increase the index to keep them unique
        down.index += len(down)

        if isinstance(filter_probs_df, pd.DataFrame):
            concat_fn = pd.concat
        else:
            concat_fn = cudf.concat

        out_df = concat_fn([filter_probs_df, down])

        assert out_df.index.is_unique

        return out_df

    assert False, "Unknown index type"


@pytest.fixture(scope="function")
def is_sliceable(index_type: typing.Literal['normal', 'skip', 'dup', 'down', 'updown']):

    return not (index_type == "dup" or index_type == "updown")


def test_count(df: cudf.DataFrame):

    meta = MessageMeta(df)

    assert meta.count == len(df)


def test_has_sliceable_index(df: cudf.DataFrame, is_sliceable: bool):

    meta = MessageMeta(df)
    assert meta.has_sliceable_index() == is_sliceable


def test_ensure_sliceable_index(df: cudf.DataFrame, is_sliceable: bool):

    meta = MessageMeta(df)

    old_index_name = meta.ensure_sliceable_index()

    assert meta.has_sliceable_index()
    assert old_index_name == (None if is_sliceable else "_index_")


def test_mutable_dataframe(df: cudf.DataFrame):

    meta = MessageMeta(df)

    with meta.mutable_dataframe() as df:
        df['v2'].iloc[3] = 47

    assert meta.copy_dataframe()['v2'].iloc[3] == 47


def test_using_ctx_outside_with_block(df: cudf.DataFrame):

    meta = MessageMeta(df)

    ctx = meta.mutable_dataframe()

    # ctx.fillna() & ctx.col
    pytest.raises(AttributeError, getattr, ctx, 'fillna')

    # ctx[col]
    pytest.raises(AttributeError, operator.getitem, ctx, 'col')

    # ctx[col] = 5
    pytest.raises(AttributeError, operator.setitem, ctx, 'col', 5)


def test_copy_dataframe(df: cudf.DataFrame):

    meta = MessageMeta(df)

    copied_df = meta.copy_dataframe()

    assert DatasetManager.assert_df_equal(copied_df, df), "Should be identical"
    assert copied_df is not df, "But should be different instances"

    # Try setting a single value on the copy
    cdf = meta.copy_dataframe()
    cdf['v2'].iloc[3] = 47
    assert DatasetManager.assert_df_equal(meta.copy_dataframe(), df), "Should be identical"
