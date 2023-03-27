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

import typing
from io import IOBase
from io import StringIO

import pandas as pd

import cudf

from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.common import write_df_to_file as write_df_to_file_cpp
from morpheus.config import CppConfig


def df_to_stream_csv(df: typing.Union[pd.DataFrame, cudf.DataFrame],
                     stream: IOBase,
                     include_header=False,
                     include_index_col=True):
    """
    Serializes a DataFrame into CSV into the provided stream object.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    stream : IOBase
        The stream where the serialized DataFrame will be written to.
    include_header : bool, optional
        Whether or not to include the header, by default False.
    include_index_col: bool, optional
        Write out the index as a column, by default True.
    """

    df.to_csv(stream, header=include_header, index=include_index_col)

    return stream


def df_to_stream_json(df: typing.Union[pd.DataFrame, cudf.DataFrame], stream: IOBase, include_index_col=True):
    """
    Serializes a DataFrame into JSON into the provided stream object.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    stream : IOBase
        The stream where the serialized DataFrame will be written to.
    include_index_col: bool, optional
        Write out the index as a column, by default True.
    """

    df.to_json(stream, orient="records", lines=True, index=include_index_col)

    return stream


def df_to_stream_parquet(df: typing.Union[pd.DataFrame, cudf.DataFrame], stream: IOBase):
    """
    Serializes a DataFrame into Parquet format into the provided stream object.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    stream : IOBase
        The stream where the serialized DataFrame will be written to.
    """

    df.to_parquet(stream)

    return stream


def df_to_csv(df: cudf.DataFrame,
              include_header=False,
              strip_newlines=False,
              include_index_col=True) -> typing.List[str]:
    """
    Serializes a DataFrame into CSV and returns the serialized output seperated by lines.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    include_header : bool, optional
        Whether or not to include the header, by default False.
    strip_newlines : bool, optional
        Whether or not to strip the newline characters from each string, by default False.
    include_index_col: bool, optional
        Write out the index as a column, by default True.

    Returns
    -------
    typing.List[str]
        List of strings for each line
    """

    str_buf = StringIO()

    df_to_stream_csv(df=df, stream=str_buf, include_header=include_header, include_index_col=include_index_col)

    # Start from beginning
    str_buf.seek(0)

    # Return list of strs to write out
    results = str_buf.readlines()
    if strip_newlines:
        results = [line.rstrip("\n") for line in results]

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

    df_to_stream_json(df=df, stream=str_buf, include_index_col=include_index_col)

    # Start from beginning
    str_buf.seek(0)

    # Return list of strs to write out
    results = str_buf.readlines()
    if strip_newlines:
        results = [line.rstrip("\n") for line in results]

    return results


def df_to_parquet(df: cudf.DataFrame, strip_newlines=False) -> typing.List[str]:
    """
    Serializes a DataFrame into Parquet and returns the serialized output seperated by lines.

    Parameters
    ----------
    df : cudf.DataFrame
        Input DataFrame to serialize.
    strip_newlines : bool, optional
        Whether or not to strip the newline characters from each string, by default False.
    Returns
    -------
    typing.List[str]
        List of strings for each line.
    """
    str_buf = StringIO()

    df_to_stream_parquet(df=df, stream=str_buf)

    # Start from beginning
    str_buf.seek(0)

    # Return list of strs to write out
    results = str_buf.readlines()
    if strip_newlines:
        results = [line.rstrip("\n") for line in results]

    return results


def write_df_to_file(df: typing.Union[pd.DataFrame, cudf.DataFrame],
                     file_name: str,
                     file_type: FileTypes = FileTypes.Auto,
                     **kwargs):
    """
    Writes the provided DataFrame into the file specified using the specified format.

    Parameters
    ----------
    df : typing.Union[pd.DataFrame, cudf.DataFrame]
        The DataFrame to serialize
    file_name : str
        The location to store the DataFrame
    file_type : FileTypes, optional
        The type of serialization to use. By default this is `FileTypes.Auto` which will determine the type from the
        filename extension
    """

    if (CppConfig.get_should_use_cpp() and isinstance(df, cudf.DataFrame)):
        # Use the C++ implementation
        return write_df_to_file_cpp(df=df, filename=file_name, file_type=file_type, **kwargs)

    mode = file_type

    if (mode == FileTypes.Auto):
        mode = determine_file_type(file_name)

    with open(file_name, mode="w") as f:

        if (mode == FileTypes.JSON):
            df_to_stream_json(df=df, stream=f, **kwargs)
        elif (mode == FileTypes.CSV):
            df_to_stream_csv(df=df, stream=f, **kwargs)
        else:
            assert False, "Unsupported filetype"
