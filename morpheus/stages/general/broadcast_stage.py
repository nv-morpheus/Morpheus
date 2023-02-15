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

import logging
import typing

import mrc
from mrc.core.node import Broadcast

from morpheus.config import Config
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class BroadcastStage(Stage):
    """
    """

    def __init__(self, c: Config, output_port_count: int = 2):

        super().__init__(c)

        self._create_ports(1, output_port_count)
        self._output_port_count = output_port_count

    @property
    def name(self) -> str:
        return "broadcast"

    def supports_cpp_node(self):
        return False

    def input_types(self) -> typing.Tuple:
        """
        Returns input type for the current stage.
        """

        return (typing.Any, )

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types.

        """
        return (typing.Any, )

    def _build(self, builder: mrc.Builder, in_stream_pairs: typing.List[StreamPair]) -> typing.List[StreamPair]:

        assert len(in_stream_pairs) == 1, "Only 1 input supported"

        in_stream_node = in_stream_pairs[0][0]
        output_type = in_stream_pairs[0][1]

        # Create a broadcast node
        broadcast_node = Broadcast(builder, "broadcast")
        builder.make_edge(in_stream_node, broadcast_node)

        if self._output_port_count <= 0:
            raise ValueError("Output port count must be greater than 0")

        out_stream_pairs = []

        count = 0
        while (count < self._output_port_count):
            out_stream_pairs.append((broadcast_node, output_type))
            count += 1

        return out_stream_pairs
