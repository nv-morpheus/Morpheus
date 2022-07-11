# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import typing

import cupy as cp

import morpheus._lib.messages as _messages
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.message_base import MessageData
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.message_meta import MessageMeta


def get_output(instance: "ResponseMemory", name: str):
    """
    Getter function used with DataClassProp for getting inference output from message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding outputs.
    name : str
        Key used to do lookup in outputs dict of message container.

    Returns
    -------
    cupy.ndarray
        Outputs corresponding to name.

    Raises
    ------
    AttributeError
        If output name does not exist in message container.

    """

    if (name not in instance.outputs):
        raise AttributeError

    return instance.outputs[name]


def set_output(instance: "ResponseMemory", name: str, value):
    """
    Setter function used with DataClassProp for setting output in message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding outputs.
    name : str
        Key used to do lookup in outputs dict of message container.
    value : cupy.ndarray
        Value to set for input.
    """

    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.outputs[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass
class ResponseMemory(MessageData, cpp_class=_messages.ResponseMemory):
    """
    Output memory block holding the results of inference.
    """
    count: int

    outputs: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)

    def get_output(self, name: str):
        if (name not in self.outputs):
            raise KeyError

        return self.outputs[name]


@dataclasses.dataclass
class ResponseMemoryProbs(ResponseMemory, cpp_class=_messages.ResponseMemoryProbs):
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)

    def __post_init__(self, probs):
        self.probs = probs


@dataclasses.dataclass
class ResponseMemoryAE(ResponseMemoryProbs, cpp_class=None):
    user_id: str = ""


@dataclasses.dataclass
class MultiResponseMessage(MultiMessage, cpp_class=_messages.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : `ResponseMemory`
        This is a response container instance for triton inference requests.
    offset : int
        Offset of each response message into the `ResponseMemory` block.
    count : int
        Inference results size of all responses.

    """
    memory: ResponseMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def outputs(self):
        """
        Get outputs stored in the ResponseMemory container.

        Returns
        -------
        cupy.ndarray
            Inference outputs.

        """

        return {key: self.get_output(key) for key in self.memory.outputs.keys()}

    def __getattr__(self, name: str) -> typing.Any:

        output_val = self.memory.outputs.get(name, None)

        if (output_val is not None):
            return output_val[self.offset:self.offset + self.count, :]

        raise AttributeError

    def get_output(self, name: str):
        """
        Get output stored in the ResponseMemory container.

        Parameters
        ----------
        name : str
            Output key name.

        Returns
        -------
        cupy.ndarray
            Inference output.

        """

        return self.memory.outputs[name][self.offset:self.offset + self.count, :]

    def copy_output_ranges(self, ranges, mask=None):
        if mask is None:
            mask = self._ranges_to_mask(self.mess_count, ranges=ranges)

        # The outputs property method returns a copy with the offsets applied
        outputs = self.outputs
        return {key: output[mask] for (key, output) in outputs.items()}

    def copy_ranges(self, ranges):
        mask = self._ranges_to_mask(self.mess_count, ranges)
        sliced_rows = self.copy_meta_ranges(ranges, mask=mask)
        sliced_count = len(sliced_rows)
        sliced_outputs = self.copy_output_ranges(ranges, mask=mask)

        mem = ResponseMemory(count=sliced_count)
        mem.outputs = sliced_outputs
        
        return MultiResponseMessage(MessageMeta(sliced_rows), 0, sliced_count, mem, 0, sliced_count)


@dataclasses.dataclass
class MultiResponseProbsMessage(MultiResponseMessage, cpp_class=_messages.MultiResponseProbsMessage):
    """
    A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
    array. Helps ensure the proper outputs are set and eases debugging.
    """

    @property
    def probs(self):
        """
        Probabilities of prediction.

        Returns
        -------
        cupy.ndarray
            probabilities

        """

        return self.get_output("probs")


@dataclasses.dataclass
class MultiResponseAEMessage(MultiResponseProbsMessage, cpp_class=None):
    """
    A stronger typed version of `MultiResponseProbsMessage` that is used for inference workloads that return a
    probability array. Helps ensure the proper outputs are set and eases debugging.
    """

    user_id: str
