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

logger = logging.getLogger(__name__)


def filter_null_data(x: DataFrameType):
    """
    Filters out null row in a dataframe's 'data' column if it exists.

    Parameters
    ----------
    x : DataFrameType
        The dataframe to fix.
    """

    if ("data" not in x):
        return x

    return x[~x['data'].isna()]


def _cudf_needs_truncate(df: cudf.DataFrame, column_max_bytes: dict[str, int]) -> bool:
    """
    Optimization, cudf contains a byte_count() method that pandas lacks.
    """
    for (col, max_bytes) in column_max_bytes.items():
        series: cudf.Series = df[col]

        assert series.dtype == 'object'

        if series.str.byte_count().max() > max_bytes:
            return True

    return False


def truncate_string_cols_by_bytes(df: DataFrameType,
                                  column_max_bytes: dict[str, int],
                                  warn_on_truncate: bool = True) -> DataFrameType:
    """
    Truncates all string columns in a dataframe to a maximum number of bytes.

    If truncation is not needed, the original dataframe is returned. If `df` is a cudf.DataFrame, and truncating is
    needed this function will convert to a pandas DataFrame to perform the truncation.

    Parameters
    ----------
    df : DataFrameType
        The dataframe to truncate.
    column_max_bytes: dict[str, int]
        A mapping of string column names to the maximum number of bytes for each column.

    Returns
    -------
    DataFrameType
        The truncated dataframe, if needed.
    """

    if isinstance(df, cudf.DataFrame):
        # cudf specific optimization
        if not _cudf_needs_truncate(df, column_max_bytes):
            return df

        # If truncating is needed we need to convert to pandas to use the str.encode() method
        df = df.to_pandas()

    for (col, max_bytes) in column_max_bytes.items():
        series: pd.Series = df[col]
        if series.dtype == 'object':
            encoded_series = series.str.encode(encoding='utf-8', errors='strict')
            if encoded_series.str.len().max() > max_bytes:
                if warn_on_truncate:
                    logger.warning("Truncating column '%s' to %d bytes", col, max_bytes)

                sliced_series = encoded_series.str.slice(0, max_bytes)

                # There is a possibility that slicing by max_len will slice a multi-byte character in half setting
                # errors='ignore' will cause the resulting string to be truncated after the last full character
                df[col] = sliced_series.str.decode(encoding='utf-8', errors='ignore')

    return df
