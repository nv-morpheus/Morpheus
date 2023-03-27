# Copyright (c) 2023, NVIDIA CORPORATION.
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

import typing

import pandas as pd

import cudf

from morpheus.messages import MessageBase
from morpheus.messages import MultiMessage


def concat_dataframes(messages: typing.List[MessageBase]) -> pd.DataFrame:
    """
    Concatinate the DataFrame associated with the collected messages into a single Pandas DataFrame.

    Parameters
    ----------
    messages : typing.List[typing.Union[MessageMeta, MultiMessage]]
        Messages containing DataFrames to concat.

    Returns
    -------
    pd.DataFrame
    """

    all_meta = []
    for x in messages:
        if isinstance(x, MultiMessage):
            df = x.get_meta()
        else:
            df = x.df

        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()

        all_meta.append(df)

    return pd.concat(all_meta)
