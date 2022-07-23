# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing
from io import StringIO

import cudf


def df_to_csv(df: cudf.DataFrame,
              include_header=False,
              strip_newline=False,
              include_index_col=True) -> typing.List[str]:
    """
    Serializes a DataFrame into CSV and returns the serialized output seperated by lines.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    include_header : bool, optional
        Whether or not to include the header, by default False.
    strip_newline : bool, optional
        Whether or not to strip the newline characters from each string, by default False.
    include_index_col: bool, optional
        Write out the index as a column, by default True.

    Returns
    -------
    typing.List[str]
        List of strings for each line
    """
    results = df.to_csv(header=include_header, index=include_index_col)
    if strip_newline:
        results = results.split("\n")
    else:
        results = [results]

    return results


def df_to_json(df: cudf.DataFrame, strip_newlines=False, include_index_col=True) -> typing.List[str]:
    """
    Serializes a DataFrame into JSON and returns the serialized output seperated by lines.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    strip_newline : bool, optional
        Whether or not to strip the newline characters from each string, by default False.
    include_index_col: bool, optional
        Write out the index as a column, by default True.
        Note: This value is currently being ignored due to a known issue in Pandas:
            https://github.com/pandas-dev/pandas/issues/37600
    Returns
    -------
    typing.List[str]
        List of strings for each line.
    """
    str_buf = StringIO()

    # Convert to list of json string objects
    df.to_json(str_buf, orient="records", lines=True, index=include_index_col)

    # Start from beginning
    str_buf.seek(0)

    # Return list of strs to write out
    results = str_buf.readlines()
    if strip_newlines:
        results = [line.rstrip("\n") for line in results]

    return results
