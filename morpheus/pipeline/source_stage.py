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

import logging
import typing
from abc import abstractmethod

import mrc

import morpheus.pipeline as _pipeline
from morpheus.config import Config
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class SourceStage(_pipeline.StreamWrapper):
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

        self._source_stream: mrc.SegmentObject = None

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
    def _build_source(self, builder: mrc.Builder) -> StreamPair:
        """
        Abstract method all derived Source classes should implement. Returns the same value as `build`.

        :meta public:

        Returns
        -------

        `morpheus.pipeline.pipeline.StreamPair`:
            A tuple containing the output `mrc.SegmentObject` object from this stage and the message data type.
        """

        pass

    @typing.final
    def _build(self, builder: mrc.Builder, in_ports_streams: typing.List[StreamPair]) -> typing.List[StreamPair]:
        # Derived source stages should override `_build_source` instead of this method. This allows for tracking the
        # True source object separate from the output stream. If any other operators need to be added after the source,
        # use `_post_build`
        assert len(self.input_ports) == 0, "Sources shouldnt have input ports"

        source_pair = self._build_source(builder)

        curr_source = source_pair[0]

        self._source_stream = curr_source

        # Now set up the output ports
        self._output_ports[0]._out_stream_pair = source_pair

        return [source_pair]

    def _post_build(self, builder: mrc.Builder, out_ports_pair: typing.List[StreamPair]) -> typing.List[StreamPair]:

        return out_ports_pair

    def _start(self):
        self._source_stream.start()

    async def join(self):
        pass
