# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""IO utilities."""

import logging

import pandas as pd

import cudf

from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import SeriesType

logger = logging.getLogger(__name__)


def filter_null_data(x: DataFrameType, column_name: str = "data") -> DataFrameType:
    """
    Filters out null row in a dataframe's 'data' column if it exists.

    Parameters
    ----------
    x : DataFrameType
        The dataframe to fix.
    column_name : str, default 'data'
        The column name to filter on.
    """

    if ("data" not in x):
        return x

    return x[~x[column_name].isna()]


def cudf_string_cols_exceed_max_bytes(df: cudf.DataFrame, column_max_bytes: dict[str, int]) -> bool:
    """
    Checks a cudf DataFrame for string columns that exceed a maximum number of bytes and thus need to be truncated by
    calling `truncate_string_cols_by_bytes`.

    This method utilizes a cudf method `Series.str.byte_count()` method that pandas lacks, which can avoid a costly
    call to truncate_string_cols_by_bytes.

    Parameters
    ----------
    df : DataFrameType
        The dataframe to check.
    column_max_bytes: dict[str, int]
        A mapping of string column names to the maximum number of bytes for each column.

    Returns
    -------
    bool
        True if truncation is needed, False otherwise.
    """
    if not isinstance(df, cudf.DataFrame):
        raise ValueError("Expected cudf DataFrame")

    for (col, max_bytes) in column_max_bytes.items():
        series: cudf.Series = df[col]

        assert series.dtype == 'object'

        if series.str.byte_count().max() > max_bytes:
            return True

    return False


def truncate_string_cols_by_bytes(df: DataFrameType,
                                  column_max_bytes: dict[str, int],
                                  warn_on_truncate: bool = True) -> bool:
    """
    Truncates all string columns in a dataframe to a maximum number of bytes. This operation is performed in-place on
    the dataframe.

    Parameters
    ----------
    df : DataFrameType
        The dataframe to truncate.
    column_max_bytes: dict[str, int]
        A mapping of string column names to the maximum number of bytes for each column.
    warn_on_truncate: bool, default True
        Whether to log a warning when truncating a column.

    Returns
    -------
    bool
        True if truncation was performed, False otherwise.
    """

    performed_truncation = False
    is_cudf = isinstance(df, cudf.DataFrame)

    for (col, max_bytes) in column_max_bytes.items():
        series: SeriesType = df[col]

        if is_cudf:
            series: pd.Series = series.to_pandas()

        assert series.dtype == 'object', f"Expected string column '{col}'"

        encoded_series = series.str.encode(encoding='utf-8', errors='strict')
        if encoded_series.str.len().max() > max_bytes:
            performed_truncation = True
            if warn_on_truncate:
                logger.warning("Truncating column '%s' to %d bytes", col, max_bytes)

            truncated_series = encoded_series.str.slice(0, max_bytes)

            # There is a possibility that slicing by max_len will slice a multi-byte character in half setting
            # errors='ignore' will cause the resulting string to be truncated after the last full character
            decoded_series = truncated_series.str.decode(encoding='utf-8', errors='ignore')

            if is_cudf:
                df[col] = cudf.Series.from_pandas(decoded_series)
            else:
                df[col] = decoded_series

    return performed_truncation
