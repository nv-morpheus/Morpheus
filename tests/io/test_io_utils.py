#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable

import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.io import utils as io_utils
from morpheus.utils.type_aliases import DataFrameType

MULTI_BYTE_STRINGS = ["ñäμɛ", "Moρφέας", "taç"]


def _mk_df(df_class: Callable[..., DataFrameType], data: dict[str, list[str]]) -> DataFrameType:
    """
    Create a dataframe with a 'data' column containing the given data, and some other columns with different data types
    """
    num_rows = len(data[list(data.keys())[0]])

    float_col = []
    int_col = []
    short_str_col = []
    for i in range(num_rows):
        float_col.append(i)
        int_col.append(i)
        short_str_col.append(f"{i}"[0:3])

    df_data = data.copy()
    df_data.update({"float_col": float_col, "int_col": int_col, "short_str_col": short_str_col})

    return df_class(df_data)


@pytest.mark.parametrize(
    "data, max_bytes, expected",
    [({
        "data": MULTI_BYTE_STRINGS[:]
    }, {
        "data": 8
    }, True), ({
        "data": MULTI_BYTE_STRINGS[:], "ignored_col": ["a" * 20, "b" * 20, "c" * 20]
    }, {
        "data": 12
    }, False), ({
        "data": MULTI_BYTE_STRINGS[:]
    }, {
        "data": 20
    }, False), ({
        "data": ["." * 20]
    }, {
        "data": 19
    }, True), ({
        "data": ["." * 20]
    }, {
        "data": 20
    }, False), ({
        "data": ["." * 20]
    }, {
        "data": 21
    }, False)])
def test_cudf_needs_truncate(data: list[str], max_bytes: int, expected: bool):
    df = _mk_df(cudf.DataFrame, data)
    assert io_utils.cudf_string_cols_exceed_max_bytes(df, max_bytes) is expected


@pytest.mark.parametrize("warn_on_truncate", [True, False])
@pytest.mark.parametrize(
    "data, max_bytes, expected_data",
    [({
        "multibyte_strings": MULTI_BYTE_STRINGS[:], "ascii_strings": ["a" * 20, "b" * 21, "c" * 19]
    }, {
        "multibyte_strings": 4, "ascii_strings": 20
    }, {
        "multibyte_strings": ["ñä", "Moρ", "taç"], "ascii_strings": ["a" * 20, "b" * 20, "c" * 19]
    }),
     ({
         "data": MULTI_BYTE_STRINGS[:], "ignored_col": ["a" * 20, "b" * 20, "c" * 20]
     }, {
         "data": 5
     }, {
         "data": ["ñä", "Moρ", "taç"], "ignored_col": ["a" * 20, "b" * 20, "c" * 20]
     }), ({
         "data": MULTI_BYTE_STRINGS[:]
     }, {
         "data": 8
     }, {
         "data": ["ñäμɛ", "Moρφέ", "taç"]
     }), ({
         "data": MULTI_BYTE_STRINGS[:]
     }, {
         "data": 9
     }, {
         "data": ["ñäμɛ", "Moρφέ", "taç"]
     }), ({
         "data": MULTI_BYTE_STRINGS[:]
     }, {
         "data": 12
     }, {
         "data": MULTI_BYTE_STRINGS[:]
     })])
def test_truncate_string_cols_by_bytes(dataset: DatasetManager,
                                       data: dict[str, list[str]],
                                       max_bytes: int,
                                       expected_data: dict[str, list[str]],
                                       warn_on_truncate: bool):
    df = _mk_df(dataset.df_class, data)

    expect_truncation = (data != expected_data)
    expected_df_class = dataset.df_class

    expected_df = _mk_df(expected_df_class, expected_data)

    performed_truncation = io_utils.truncate_string_cols_by_bytes(df, max_bytes, warn_on_truncate=warn_on_truncate)

    assert performed_truncation is expect_truncation
    assert isinstance(df, expected_df_class)

    dataset.assert_df_equal(df, expected_df)
