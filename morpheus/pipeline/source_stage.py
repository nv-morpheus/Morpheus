# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import asyncio
import collections
import inspect
import logging
import os
import signal
import time
import typing
from abc import ABC
from abc import abstractmethod

import neo
import networkx
import typing_utils
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.messages import MultiMessage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.pipeline.stream_wrapper import StreamWrapper
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)

class SourceStage(StreamWrapper):
    """
    The SourceStage is mandatory for the Morpheus pipeline to run. This stage represents the start of the pipeline. All
    `SourceStage` object take no input but generate output.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        super().__init__(c)

        self._start_callbacks: typing.List[typing.Callable] = []
        self._stop_callbacks: typing.List[typing.Callable] = []

        self._source_stream: neo.Node = None

    @property
    def input_count(self) -> int:
        """
        Return None for no max intput count.

        Returns
        -------
        int
            Input count.

        """
        return None

    @abstractmethod
    def _build_source(self, seg: neo.Segment) -> StreamPair:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`.

        :meta public:

        Returns
        -------

        `morpheus.pipeline.pipeline.StreamPair`:
            A tuple containing the output `neo.Node` object from this stage and the message data type.
        """

        pass

    @typing.final
    def _build(self, seg: neo.Segment, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 0, "Sources shouldnt have input ports"

        source_pair = self._build_source(seg)

        curr_source = source_pair[0]

        self._source_stream = curr_source

        # Now setup the output ports
        self._output_ports[0]._out_stream_pair = source_pair

        return [source_pair]

    def _post_build(self, seg: neo.Segment, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        return out_ports_pair

    def _start(self):
        self._source_stream.start()

    def stop(self):
        self._source_stream.stop()

    async def join(self):
        self._source_stream.join()
