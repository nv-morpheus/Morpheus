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

import pandas as pd
import pytest

import cudf

from _utils.dataset_manager import DatasetManager
from morpheus.io import utils as io_utils
from morpheus.utils.type_aliases import DataFrameType

MULTI_BYTE_STRINGS = ["ñäμɛ", "Moρφευσ", "río"]


def _mk_df(df_class: Callable[..., DataFrameType], data: list[str]) -> DataFrameType:
    """
    Create a dataframe with a 'data' column containing the given data, and some other columns with different data types
    """
    float_col = []
    int_col = []
    short_str_col = []
    for i in range(len(data)):
        float_col.append(i)
        int_col.append(i)
        short_str_col.append(f"{i}"[0:3])

    return df_class({"data": data, "float_col": float_col, "int_col": int_col, "short_str_col": short_str_col})


@pytest.mark.parametrize("data, max_bytes, expected",
                         [(MULTI_BYTE_STRINGS[:], 8, True), (MULTI_BYTE_STRINGS[:], 12, False),
                          (MULTI_BYTE_STRINGS[:], 20, False), (["." * 20], 19, True), (["." * 20], 20, False),
                          (["." * 20], 21, False)])
def test_cudf_needs_truncate(data: list[str], max_bytes: int, expected: bool):
    df = _mk_df(cudf.DataFrame, data)
    assert io_utils._cudf_needs_truncate(df, max_bytes) is expected


@pytest.mark.parametrize("warn_on_truncate", [True, False])
@pytest.mark.parametrize(
    "data, max_bytes, expected_data",
    [(MULTI_BYTE_STRINGS[:], 4, ["ñä", "Moρ", "río"]), (MULTI_BYTE_STRINGS[:], 5, ["ñä", "Moρ", "río"]),
     (MULTI_BYTE_STRINGS[:], 8, ["ñäμɛ", "Moρφε", "río"]), (MULTI_BYTE_STRINGS[:], 9, ["ñäμɛ", "Moρφε", "río"]),
     (MULTI_BYTE_STRINGS[:], 12, MULTI_BYTE_STRINGS[:]), (MULTI_BYTE_STRINGS[:], 20, MULTI_BYTE_STRINGS[:]),
     (["." * 20], 19, ["." * 19]), (["." * 20], 20, ["." * 20]), (["." * 20], 21, ["." * 20])])
def test_truncate_string_cols_by_bytes(dataset: DatasetManager,
                                       data: list[str],
                                       max_bytes: int,
                                       expected_data: list[str],
                                       warn_on_truncate: bool):
    input_df = _mk_df(dataset.df_class, data)

    if data == expected_data:
        expected_df_class = dataset.df_class
    else:
        expected_df_class = pd.DataFrame

    expected_df = _mk_df(expected_df_class, expected_data)

    result_df = io_utils.truncate_string_cols_by_bytes(input_df, max_bytes, warn_on_truncate=warn_on_truncate)

    assert isinstance(result_df, expected_df_class)
    dataset.assert_df_equal(result_df, expected_df)
