# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pandas as pd

import cudf

from morpheus.common import FileTypes
from morpheus.common import determine_file_type
from morpheus.io.utils import filter_null_data


def cudf_json_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. When loading a JSON file, cuDF converts ``\\n`` to
    ``\\\\n`` for some reason.
    """
    if ("data" in x and not x.empty):
        x["data"] = x["data"].str.replace('\\n', '\n', regex=False)

    return x


def read_file_to_df(file_name: str,
                    file_type: FileTypes,
                    parser_kwargs: dict = {},
                    filter_nulls: bool = True,
                    df_type: typing.Literal["cudf", "pandas"] = "pandas") -> typing.Union[cudf.DataFrame, pd.DataFrame]:
    """
    Reads a file into a dataframe and performs any of the necessary cleanup.

    Parameters
    ----------
    file_name : str
        File to read.
    file_type : `morpheus.common.FileTypes`
        Type of file. Leave as Auto to determine from the extension.
    parser_kwargs : dict, optional
        Any argument to pass onto the parse, by default {}.
    filter_nulls : bool, optional
        Whether to filter null rows after loading, by default True.
    df_type : typing.Literal[, optional
        What type of parser to use. Options are 'cudf' and 'pandas', by default "pandas".

    Returns
    -------
    typing.Union[cudf.DataFrame, pandas.DataFrame]
        A parsed DataFrame.
    """

    mode = file_type

    if (mode == FileTypes.Auto):
        mode = determine_file_type(file_name)

    # Special args for JSON
    kwargs = {}
    if (mode == FileTypes.JSON):
        kwargs["lines"] = True

    # Update with any args set by the user. User values overwrite defaults
    kwargs.update(parser_kwargs)

    df_class = cudf if df_type == "cudf" else pd

    if (mode == FileTypes.JSON):
        df = df_class.read_json(file_name, **kwargs)

        if (filter_nulls):
            df = filter_null_data(df)

        if (df_type == "cudf"):
            df = cudf_json_onread_cleanup(df)

        return df
    elif (mode == FileTypes.CSV):
        df: pd.DataFrame = df_class.read_csv(file_name, **kwargs)

        if (len(df.columns) > 1 and df.columns[0] == "Unnamed: 0" and df.iloc[:, 0].dtype == cudf.dtype(int)):
            df.set_index("Unnamed: 0", drop=True, inplace=True)
            df.index.name = ""
            df.sort_index(inplace=True)

        if (filter_nulls):
            df = filter_null_data(df)

        return df
    else:
        assert False, "Unsupported file type mode: {}".format(mode)
