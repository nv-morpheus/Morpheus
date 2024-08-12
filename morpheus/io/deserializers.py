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
"""DataFrame deserializers."""

import io
import typing

import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    import cudf

from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.common import read_file_to_df as read_file_to_df_cpp
from morpheus.config import CppConfig
from morpheus.io.utils import filter_null_data
from morpheus.utils.type_aliases import DataFrameType


def _read_file_to_df_py(*,
                        file_name: typing.Union[str, io.IOBase],
                        file_type: FileTypes,
                        parser_kwargs: dict,
                        df_type: typing.Literal["cudf", "pandas"]) -> DataFrameType:
    if (parser_kwargs is None):
        parser_kwargs = {}

    mode = file_type

    if (mode == FileTypes.Auto):
        # The DFPFileToDataFrameStage passes an instance of an fsspec file opener instead of a filename to this method.
        # The opener objects are subclasses of io.IOBase, which avoids introducing fsspec to this part of the API
        if (isinstance(file_name, io.IOBase)):
            if (hasattr(file_name, 'path')):  # This attr is not in the base
                filepath = file_name.path
            else:
                raise ValueError("Unable to determine file type from instance of io.IOBase,"
                                 " set `file_type` to a value other than Auto")
        else:
            filepath = file_name
        mode = determine_file_type(filepath)

    # Special args for JSON
    kwargs = {}
    if (mode == FileTypes.JSON):
        kwargs["lines"] = True

    # Update with any args set by the user. User values overwrite defaults
    kwargs.update(parser_kwargs)

    if df_type == "cudf":
        import cudf
        df_class = cudf
    else:
        df_class = pd

    df = None
    if (mode == FileTypes.JSON):
        df = df_class.read_json(file_name, **kwargs)

    elif (mode == FileTypes.CSV):
        df: DataFrameType = df_class.read_csv(file_name, **kwargs)

        if (len(df.columns) > 1 and df.columns[0] == "Unnamed: 0" and df.iloc[:, 0].dtype == np.dtype(int)):
            df.set_index("Unnamed: 0", drop=True, inplace=True)
            df.index.name = ""
            df.sort_index(inplace=True)

    elif (mode == FileTypes.PARQUET):
        df = df_class.read_parquet(file_name, **kwargs)

    else:
        assert False, f"Unsupported file type mode: {mode}"

    assert df is not None

    return df


def read_file_to_df(file_name: typing.Union[str, io.IOBase],
                    file_type: FileTypes = FileTypes.Auto,
                    parser_kwargs: dict = None,
                    filter_nulls: bool = True,
                    filter_null_columns: list[str] | str = 'data',
                    df_type: typing.Literal["cudf", "pandas"] = "pandas") -> DataFrameType:
    """
    Reads a file into a dataframe and performs any of the necessary cleanup.

    Parameters
    ----------
    file_name : str
        File to read.
    file_type : `morpheus.common.FileTypes`
        Type of file. Leave as Auto to determine from the extension.
    parser_kwargs : dict, optional
        Any argument to pass onto the parse, by default {}. Ignored when C++ execution is enabled and `df_type="cudf"`
    filter_nulls : bool, optional
        Whether to filter null rows after loading, by default True.
    filter_null_columns : list[str]|str, default = 'data'
        Column or columns to filter null values from. Ignored when `filter_null` is False.
    df_type : typing.Literal[, optional
        What type of parser to use. Options are 'cudf' and 'pandas', by default "pandas".

    Returns
    -------
    DataFrameType
        A parsed DataFrame.
    """

    # The C++ reader only supports cudf dataframes
    if (CppConfig.get_should_use_cpp() and df_type == "cudf"):
        df = read_file_to_df_cpp(file_name, file_type)
    else:
        df = _read_file_to_df_py(file_name=file_name, file_type=file_type, parser_kwargs=parser_kwargs, df_type=df_type)

    if (filter_nulls):
        if isinstance(filter_null_columns, str):
            filter_null_columns = [filter_null_columns]

        for col in filter_null_columns:
            df = filter_null_data(df, column_name=col)

    return df
