# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.tensor_memory import TensorMemory


@dataclasses.dataclass
class InferenceMemory(TensorMemory, cpp_class=_messages.InferenceMemory):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in cupy arrays.
    """


def get_input(instance, name: str):
    """
    Getter function used with DataClassProp for getting inference input from message containers derived
    from InferenceMemory.

    Parameters
    ----------
    instance : `InferenceMemory`
        Message container holding inputs.
    name : str
        Key used to do lookup in inputs dict of message container.

    Returns
    -------
    cupy.ndarray
        Inputs corresponding to name.

    Raises
    ------
    AttributeError
        If input name does not exist in message container.
    """
    if (name not in instance.tensors):
        raise AttributeError

    return instance.tensors[name]


def set_input(instance, name: str, value):
    """
    Setter function used with DataClassProp for setting inference input in message containers derived
    from InferenceMemory.

    Parameters
    ----------
    instance : `InferenceMemory`
        Message container holding inputs.
    name : str
        Key used to do lookup in inputs dict of message container.
    value : cupy.ndarray
        Value to set for input.
    """
    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.tensors[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass(init=False)
class InferenceMemoryNLP(InferenceMemory, cpp_class=_messages.InferenceMemoryNLP):
    """
    This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.

    Parameters
    ----------
    input_ids : cupy.ndarray
        The token-ids for each string padded with 0s to max_length.
    input_mask : cupy.ndarray
        The mask for token-ids result where corresponding positions identify valid token-id values.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    input_mask: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __init__(self, input_ids, input_mask, seq_ids):
        super().__init__(count=len(input_ids),
                         tensors={
                             'input_ids': input_ids, 'input_mask': input_mask, 'seq_ids': seq_ids
                         })


@dataclasses.dataclass(init=False)
class InferenceMemoryFIL(InferenceMemory, cpp_class=_messages.InferenceMemoryFIL):
    """
    This is a container class for data that needs to be submitted to the inference server for FIL category
    usecases.

    Parameters
    ----------
    input__0 : cupy.ndarray
        Inference input.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input__0: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __init__(self, input__0, seq_ids):
        super().__init__(count=len(input__0), tensors={'input__0': input__0, 'seq_ids': seq_ids})


@dataclasses.dataclass(init=False)
class InferenceMemoryAE(InferenceMemory, cpp_class=None):
    """
    This is a container class for data that needs to be submitted to the inference server for FIL category
    usecases.

    Parameters
    ----------
    input__0 : cupy.ndarray
        Inference input.
    seq_ids : cupy.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e., if some messages get broken into multiple inference requests).

    """
    input: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __init__(self, input__0, seq_ids):
        super().__init__(count=len(input__0), tensors={'input__0': input__0, 'seq_ids': seq_ids})


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage, cpp_class=_messages.MultiInferenceMessage):
    """
    This is a container class that holds the TensorMemory container and the metadata of the data contained
    within it. Builds on top of the `MultiMessage` class to add additional data for inferencing.

    This class requires two separate memory blocks for a batch. One for the message metadata (i.e., start time,
    IP address, etc.) and another for the raw inference inputs (i.e., input_ids, seq_ids). Since there can be
    more inference input requests than messages (This happens when some messages get broken into multiple
    inference requests) this class stores two different offset and count values. `mess_offset` and
    `mess_count` refer to the offset and count in the message metadata batch and `offset` and `count` index
    into the inference batch data.

    Parameters
    ----------
    memory : `TensorMemory`
        Inference memory.
    offset : int
        Message offset in inference memory instance.
    count : int
        Message count in inference memory instance.

    """
    memory: TensorMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def inputs(self):
        """
        Get inputs stored in the TensorMemory container.

        Returns
        -------
        cupy.ndarray
            Inference inputs.

        """

        return {key: self.get_input(key) for key in self.memory.tensors.keys()}

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, name: str) -> typing.Any:

        input_val = self.memory.tensors.get(name, None)

        if (input_val is not None):
            return input_val[self.offset:self.offset + self.count, :]

        raise AttributeError

    def get_input(self, name: str):
        """
        Get input stored in the TensorMemory container.

        Parameters
        ----------
        name : str
            Input key name.

        Returns
        -------
        cupy.ndarray
            Inference input.

        """

        return self.memory.tensors[name][self.offset:self.offset + self.count, :]

    def get_slice(self, start, stop):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.

        Returns
        -------
        `MultiInferenceMessage`
            A new `MultiInferenceMessage` with sliced offset and count.

        """
        mess_start = self.mess_offset + self.seq_ids[start, 0].item()
        mess_stop = self.mess_offset + self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceMessage(meta=self.meta,
                                     mess_offset=mess_start,
                                     mess_count=mess_stop - mess_start,
                                     memory=self.memory,
                                     offset=start,
                                     count=stop - start)


@dataclasses.dataclass
class MultiInferenceNLPMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceNLPMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    @property
    def input_ids(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self.get_input("input_ids")

    @property
    def input_mask(self):
        """
        Returns mask for token-ids result where corresponding positions identify valid token-id values.

        Returns
        -------
        cupy.ndarray
            The mask for token-ids result where corresponding positions identify valid token-id values.

        """

        return self.get_input("input_mask")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of which inference requests belong to each message.

        Returns
        -------
        cupy.ndarray
            Ids used to index from an inference input to a message. Necessary since there can be more
            inference inputs than messages (i.e., if some messages get broken into multiple inference requests).

        """

        return self.get_input("seq_ids")


@dataclasses.dataclass
class MultiInferenceFILMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceFILMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    @property
    def input__0(self):
        """
        Input to FIL model inference.

        Returns
        -------
        cupy.ndarray
            Input data.

        """

        return self.get_input("input__0")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            Sequence ids.

        """

        return self.get_input("seq_ids")
