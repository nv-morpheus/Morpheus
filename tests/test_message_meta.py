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

from _utils.dataset_manager import DatasetManager
# pylint: disable=morpheus-incorrect-lib-from-import
from morpheus._lib.messages import MessageMeta as MessageMetaCpp
from morpheus.messages.message_meta import MessageMeta


@pytest.fixture(name="index_type", scope="function", params=["normal", "skip", "dup", "down", "updown"])
def fixture_index_type(request: pytest.FixtureRequest) -> typing.Literal["normal", "skip", "dup", "down", "updown"]:
    return request.param


@pytest.fixture(name="df", scope="function")
def fixture_df(
    use_cpp: bool,  # pylint: disable=unused-argument
    dataset: DatasetManager,
    index_type: typing.Literal['normal', 'skip', 'dup', 'down',
                               'updown']) -> typing.Union[cudf.DataFrame, pd.DataFrame]:
    filter_probs_df = dataset["filter_probs.csv"]

    if (index_type == "normal"):
        return filter_probs_df

    if (index_type == "skip"):
        # Skip some rows
        return filter_probs_df.iloc[::3, :].copy()

    if (index_type == "dup"):
        # Duplicate
        return dataset.dup_index(filter_probs_df, count=2)

    if (index_type == "down"):
        # Reverse
        return filter_probs_df.iloc[::-1, :].copy()

    if (index_type == "updown"):
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


@pytest.fixture(name="is_sliceable", scope="function")
def fixture_is_sliceable(index_type: typing.Literal['normal', 'skip', 'dup', 'down', 'updown']):

    return index_type not in ("dup", "updown")


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

    with meta.mutable_dataframe() as df_:
        df_['v2'].iloc[3] = 47

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

    DatasetManager.assert_df_equal(copied_df, df, assert_msg="Should be identical")
    assert copied_df is not df, "But should be different instances"

    # Try setting a single value on the copy
    cdf = meta.copy_dataframe()
    cdf['v2'].iloc[3] = 47
    DatasetManager.assert_df_equal(meta.copy_dataframe(), df, assert_msg="Should be identical")


@pytest.mark.use_cpp
def test_pandas_df_cpp(dataset_pandas: DatasetManager):
    """
    Test for issue #821, calling the `df` property returns an empty cudf dataframe.
    """
    df = dataset_pandas["filter_probs.csv"]
    assert isinstance(df, pd.DataFrame)

    meta = MessageMeta(df)
    assert isinstance(meta, MessageMetaCpp)
    assert isinstance(meta.df, cudf.DataFrame)
    DatasetManager.assert_compare_df(meta.df, df)
