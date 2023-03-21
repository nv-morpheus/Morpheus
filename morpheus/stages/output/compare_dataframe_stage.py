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

from morpheus._lib.common import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MessageMeta
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.utils import compare_df


@register_stage("compare-df")
class CompareDataframeStage(InMemorySinkStage):
    """
    Collects incoming messages, comparing the concatinated dataframe of all messages against an expected dataframe
    `compare_df`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    compare_df : typing.Union[pd.DataFrame, str]
        Dataframe to compare against the aggregate Dataframe composed from the received messages. When `compare_df` is
        a string it is assumed to be a file path.
    """

    def __init__(self, c: Config, compare_df: typing.Union[pd.DataFrame, str]):
        super().__init__(c)

        if isinstance(compare_df, str):
            compare_df = read_file_to_df(compare_df, file_type=FileTypes.Auto, df_type='pandas')

        self._compare_df = compare_df

    @property
    def name(self) -> str:
        return "compare"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`morpheus.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, )

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

        return compare_df.compare_df(self._compare_df, combined_df)
