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

import mrc
import pandas as pd

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


class InMemorySinkStage(SinglePortStage):
    """
    Collects incoming messages into a list.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._messages = []

    @property
    def name(self) -> str:
        return "to-mem"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`morpheus.messages.MessageMeta`, )
            Accepted input types.

        """
        return (MessageMeta, MultiMessage)

    def supports_cpp_node(self) -> bool:
        return False

    def clear(self):
        """
        Clear the messages that have been collected so far
        """
        self._messages.clear()

    def get_messages(self) -> typing.List[typing.Union[MessageMeta, MultiMessage]]:
        """
        Returns
        -------
        list
            Results collected messages
        """
        return self._messages

    def concat_dataframes(self, clear=True) -> pd.DataFrame:
        """
        Concatinate the DataFrame associated with the collected messages into a single Pandas DataFrame,
        and optionally clearing the list of collected messages.

        Returns
        -------
        pd.DataFrame
        """
        all_meta = []
        for x in self._messages:
            if isinstance(x, MultiMessage):
                df = x.get_meta()
            else:
                df = x.df

            if isinstance(df, cudf.DataFrame):
                df = df.to_pandas()

            all_meta.append(df)

        df = pd.concat(all_meta)

        if clear:
            self.clear()

        return df

    def _append_message(self, message: typing.Union[MessageMeta,
                                                    MultiMessage]) -> typing.Union[MessageMeta, MultiMessage]:
        self._messages.append(message)
        return message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self._append_message)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
