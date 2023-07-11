# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import typing

import pandas as pd
from nvtabular import ColumnSelector

import cudf


def json_flatten(col_selector: ColumnSelector,
                 df: typing.Union[pd.DataFrame, cudf.DataFrame]) -> typing.Union[pd.DataFrame, cudf.DataFrame]:
    """
    Flattens JSON columns in the given DataFrame and concatenates them into a single DataFrame.

    Parameters
    ----------
    col_selector : ColumnSelector
        An instance of ColumnSelector that contains the names of the columns to flatten.
    df : Union[pd.DataFrame, cudf.DataFrame]
        The input DataFrame that contains the JSON columns to flatten.

    Returns
    -------
    Union[pd.DataFrame, cudf.DataFrame]
        A new DataFrame with flattened JSON columns. If 'df' was a cudf.DataFrame,
        the return type is cudf.DataFrame. Otherwise, it is pd.DataFrame.
    """
    convert_to_cudf = False
    if isinstance(df, cudf.DataFrame):
        convert_to_cudf = True

    # Normalize JSON columns and accumulate into a single dataframe
    df_normalized = None
    for col in col_selector.names:
        pd_series = df[col] if not convert_to_cudf else df[col].to_pandas()
        pd_series = pd_series.apply(lambda x: x if isinstance(x, dict) else json.loads(x))
        pdf_norm = pd.json_normalize(pd_series)
        pdf_norm.rename(columns=lambda x, col=col: col + "." + x, inplace=True)
        pdf_norm.reset_index(drop=True, inplace=True)

        if (df_normalized is None):
            df_normalized = pdf_norm
        else:
            df_normalized = pd.concat([df_normalized, pdf_norm], axis=1)

    # Convert back to cudf if necessary
    if convert_to_cudf:
        df_normalized = cudf.from_pandas(df_normalized)

    return df_normalized
