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

import logging
import typing

import morpheus.pipeline as _pipeline
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class Sender():
    """
    The `Sender` object represents a port on a `StreamWrapper` object that sends messages to a `Receiver`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StreamWrapper`
            Parent `StreamWrapper` object.
        port_number : int
            Sender port number.
    """

    def __init__(self, parent: "_pipeline.StreamWrapper", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._output_receivers: typing.List[_pipeline.Receiver] = []

        self._out_stream_pair: StreamPair = (None, None)

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        # Sender is complete when the type or stream has been set
        return self._out_stream_pair != (None, None)

    @property
    def out_pair(self):
        return self._out_stream_pair

    @property
    def out_stream(self):
        return self._out_stream_pair[0]

    @property
    def out_type(self):
        return self._out_stream_pair[1]
