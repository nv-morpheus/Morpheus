# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("dropna", modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER])
class DropNullStage(SinglePortStage):
    """
    Drop null data entries from a DataFrame.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    column : str
        Column name to perform null check.

    """

    def __init__(self, c: Config, column: str):
        super().__init__(c)

        self._column = column

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "dropna"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (MessageMeta, )

    def supports_cpp_node(self):
        # Enable support by default
        return False

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        def on_next(x: MessageMeta):

            y = MessageMeta(x.df[~x.df[self._column].isna()])

            return y

        node = builder.make_node(self.unique_name, ops.map(on_next), ops.filter(lambda x: not x.df.empty))
        builder.make_edge(stream, node)
        stream = node

        return stream, input_stream[1]
