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
from morpheus.pipeline.stream_wrapper import StreamWrapper
from morpheus.utils.atomic_integer import AtomicInteger
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import greatest_ancestor
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)

class Receiver():
    """
    The `Receiver` object represents a downstream port on a `StreamWrapper` object that gets messages from a `Sender`.

    Parameters
        ----------
        parent : `morpheus.pipeline.pipeline.StreamWrapper`
            Parent `StreamWrapper` object.
        port_number : int
            Receiver port number.
    """

    def __init__(self, parent: "StreamWrapper", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._is_linked = False

        self._input_type = None
        self._input_stream = None

        self._input_senders: typing.List[Sender] = []

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        """
        A receiver is complete if all input senders are also complete.
        """
        return all([x.is_complete for x in self._input_senders])

    @property
    def is_partial(self):
        """
        A receiver is partially complete if any input sender is complete. Receivers are usually partially complete if
        there is a circular pipeline.
        """
        # Its partially complete if any input sender is complete
        return any([x.is_complete for x in self._input_senders])

    @property
    def in_pair(self):
        return (self.in_stream, self.in_pair)

    @property
    def in_stream(self):
        return self._input_stream

    @property
    def in_type(self):
        return self._input_type

    def get_input_pair(self) -> StreamPair:

        assert self.is_partial, "Must be partially complete to get the input pair!"

        # Build the input from the senders
        if (self._input_stream is None and self._input_type is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input stream/type is determined from the sole Sender
                sender = self._input_senders[0]

                self._input_stream = sender.out_stream
                self._input_type = sender.out_type
                self._is_linked = True
            else:
                # We have multiple senders. Create a dummy stream to connect all senders
                if (self.is_complete):
                    # Connect all streams now
                    # self._input_stream = streamz.Stream(upstreams=[x.out_stream for x in self._input_senders],
                    #                                     asynchronous=True,
                    #                                     loop=IOLoop.current())
                    raise NotImplementedError("Still using streamz")
                    self._is_linked = True
                else:
                    # Create a dummy stream that needs to be linked later
                    # self._input_stream = streamz.Stream(asynchronous=True, loop=IOLoop.current())
                    raise NotImplementedError("Still using streamz")

                # Now determine the output type from what we have
                great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

                if (great_ancestor is None):
                    # TODO: Add stage, port, and type info to message
                    raise RuntimeError(("Cannot determine single type for senders of input port. "
                                        "Use a merge stage to handle different types of inputs."))

                self._input_type = great_ancestor

        return (self._input_stream, self._input_type)

    def link(self):
        """
        The linking phase determines the final type of the `Receiver` and connects all underlying stages.

        Raises:
            RuntimeError: Throws a `RuntimeError` if the predicted input port type determined during the build phase is
            different than the current port type.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_linked):
            return

        # Check that the types still work
        great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

        if (not typing_utils.issubtype(great_ancestor, self._input_type)):
            # TODO: Add stage, port, and type info to message
            raise RuntimeError(
                "Invalid linking phase. Input port type does not match predicted type determined during build phase")

        for out_stream in [x.out_stream for x in self._input_senders]:
            out_stream.connect(self._input_stream)

        self._is_linked = True
