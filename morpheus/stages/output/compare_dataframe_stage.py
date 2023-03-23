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

import copy
import typing

import pandas as pd

import cudf

from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils import compare_df


class CompareDataFrameStage(InMemorySinkStage):
    """
    Collects incoming messages, comparing the concatinated dataframe of all messages against an expected dataframe
    `compare_df`.

    If a column name is matched by both `include` and `exclude`, it will be excluded.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    compare_df : typing.Union[pd.DataFrame, cudf.DataFrame, str]
        Dataframe to compare against the aggregate DataFrame composed from the received messages. When `compare_df` is
        a string it is assumed to be a file path.
    include : typing.List[str], optional
        List of regex patterns to match columns to include. By default all columns are included.
    exclude : typing.List[str], optional
        List of regex patters to match columns to exclude. By default no columns are excluded.
    index_col : str, optional
        Whether to convert any column in the dataset to the index. Useful when the pipeline will change the index,
        by default None.
    abs_tol : float, default = 0.001
        Absolute tolerance to use when comparing float columns.
    rel_tol : float, default = 0.05
        Relative tolerance to use when comparing float columns.
    reset_index : bool, default = False
        When `True` the index of the aggregate DataFrame will be reset. Useful for testing with KafkaSourceStage,
        where the index restarts for each `MessageMeta` emitted.
    """

    def __init__(self,
                 c: Config,
                 compare_df: typing.Union[pd.DataFrame, cudf.DataFrame, str],
                 include: typing.List[str] = None,
                 exclude: typing.List[str] = None,
                 index_col: str = None,
                 abs_tol: float = 0.001,
                 rel_tol: float = 0.005,
                 reset_index=False):
        super().__init__(c)

        if isinstance(compare_df, str):
            compare_df = read_file_to_df(compare_df, df_type='pandas')
        elif isinstance(compare_df, cudf.DataFrame):
            compare_df = compare_df.to_pandas()

        self._compare_df = compare_df

        # Make copies of the arrays to prevent changes after the Regex is compiled
        self._include_columns = copy.copy(include)
        self._exclude_columns = copy.copy(exclude)
        self._index_col = index_col
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol
        self._reset_index = reset_index

    @property
    def name(self) -> str:
        return "compare"

    def supports_cpp_node(self) -> bool:
        return False

    def get_results(self, clear=True) -> dict:
        """
        Returns the results dictionary. This is the same dictionary that is written to results_file_name

        Returns
        -------
        dict
            Results dictionary
        """
        if (len(self._messages) == 0):
            return {}

        combined_df = self.concat_dataframes(clear=clear)

        if self._reset_index:
            combined_df.reset_index(inplace=True)

        return compare_df.compare_df(self._compare_df,
                                     combined_df,
                                     include_columns=self._include_columns,
                                     exclude_columns=self._exclude_columns,
                                     replace_idx=self._index_col,
                                     abs_tol=self._abs_tol,
                                     rel_tol=self._rel_tol)
