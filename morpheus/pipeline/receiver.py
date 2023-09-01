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

import mrc
import typing_utils

import morpheus.pipeline as _pipeline
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils.type_utils import greatest_ancestor

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

    def __init__(self, parent: "_pipeline.StreamWrapper", port_number: int):

        self._parent = parent
        self.port_number = port_number

        self._is_type_linked = False
        self._is_node_linked = False

        self._input_type: type = None
        self._input_node: mrc.SegmentObject = None

        self._input_senders: typing.List[_pipeline.Sender] = []

    @property
    def parent(self):
        return self._parent

    @property
    def is_complete(self):
        """
        A receiver is complete if all input senders are also complete.
        """
        return all(x.is_complete for x in self._input_senders)

    @property
    def is_partial(self):
        """
        A receiver is partially complete if any input sender is complete. Receivers are usually partially complete if
        there is a circular pipeline.
        """
        # Its partially complete if any input sender is complete
        return any(x.is_complete for x in self._input_senders)

    @property
    def in_pair(self):
        return (self.in_node, self.in_pair)

    @property
    def in_node(self):
        return self._input_node

    @property
    def in_type(self):
        return self._input_type

    def get_input_pair(self, builder: mrc.Builder) -> StreamPair:
        """
        Returns the input `StreamPair` which is a tuple consisting of the parent node and the parent node's output type.
        """

        assert self.is_partial, "Must be partially complete to get the input pair!"

        # Build the input from the senders
        if (self._input_node is None and self._input_type is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input stream/type is determined from the sole Sender
                sender = self._input_senders[0]

                self._input_node = sender.out_node
                self._input_type = sender.out_type
                self._is_type_linked = True
            else:
                # We have multiple senders. Create a dummy stream to connect all senders
                self._input_node = builder.make_node_component(
                    f"{self.parent.unique_name}-reciever[{self.port_number}]", mrc.core.operators.map(lambda x: x))

                if (self.is_complete):
                    # Connect all streams now
                    for input_sender in self._input_senders:
                        builder.make_edge(input_sender.out_node, self._input_node)

                    self._is_type_linked = True

                # Now determine the output type from what we have
                great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

                if (great_ancestor is None):
                    # TODO(MDD): Add stage, port, and type info to message
                    raise RuntimeError(("Cannot determine single type for senders of input port. "
                                        "Use a merge stage to handle different types of inputs."))

                self._input_type = great_ancestor

        return (self._input_node, self._input_type)

    def get_input_type(self) -> type:
        """
        Returns the the parent node's output type.
        """

        assert self.is_partial, "Must be partially complete to get the input type!"

        # Build the input from the senders
        if (self._input_type is None):
            # First check if we only have 1 input sender
            if (len(self._input_senders) == 1):
                # In this case, our input stream/type is determined from the sole Sender
                sender = self._input_senders[0]
                self._input_type = sender.out_type
                self._is_type_linked = True  #???
            else:
                # TODO: get_input_pair would use the mrc builder to create a dummy stream to connect all senders
                # Not sure if we need to do that

                # Now determine the output type from what we have
                great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

                if (great_ancestor is None):
                    # TODO(MDD): Add stage, port, and type info to message
                    raise RuntimeError(("Cannot determine single type for senders of input port. "
                                        "Use a merge stage to handle different types of inputs."))

                self._input_type = great_ancestor

        return self._input_type

    def link_type(self):
        """
        The linking phase determines the final type of the `Receiver` and connects all underlying stages.

        Raises:
            RuntimeError: Throws a `RuntimeError` if the predicted input port type determined during the build phase is
            different than the current port type.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_type_linked):
            return

        # Check that the types still work
        great_ancestor = greatest_ancestor(*[x.out_type for x in self._input_senders if x.is_complete])

        if (not typing_utils.issubtype(great_ancestor, self._input_type)):
            raise RuntimeError(f"Input port type {great_ancestor} does not match {self._input_type} for {self._parent}")

        self._is_type_linked = True

    def link_node(self, builder: mrc.Builder):
        """
        The linking phase determines the final type of the `Receiver` and connects all underlying stages.

        Raises:
            RuntimeError: Throws a `RuntimeError` if the predicted input port type determined during the build phase is
            different than the current port type.
        """

        assert self.is_complete, "Must be complete before linking!"

        if (self._is_node_linked):
            return

        for out_stream in [x.out_stream for x in self._input_senders]:
            builder.make_edge(out_stream, self._input_node)

        self._is_node_linked = True
