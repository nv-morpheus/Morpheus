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

import morpheus._lib.messages as _messages
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_tensor_message import MultiTensorMessage


@dataclasses.dataclass
class MultiInferenceMessage(MultiTensorMessage, cpp_class=_messages.MultiInferenceMessage):
    """
    This is a container class that holds the InferenceMemory container and the metadata of the data contained
    within it. Builds on top of the `MultiTensorMessage` class to add additional data for inferencing.

    This class requires two separate memory blocks for a batch. One for the message metadata (i.e., start time,
    IP address, etc.) and another for the raw inference inputs (i.e., input_ids, seq_ids). Since there can be
    more inference input requests than messages (This happens when some messages get broken into multiple
    inference requests) this class stores two different offset and count values. `mess_offset` and
    `mess_count` refer to the offset and count in the message metadata batch and `offset` and `count` index
    into the inference batch data.
    """

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count)

    @property
    def inputs(self):
        """
        Get inputs stored in the InferenceMemory container. Alias for `MultiInferenceMessage.tensors`.

        Returns
        -------
        cupy.ndarray
            Inference inputs.

        """
        return self.tensors

    def get_input(self, name: str):
        """
        Get input stored in the InferenceMemory container. Alias for `MultiInferenceMessage.get_tensor`.

        Parameters
        ----------
        name : str
            Input key name.

        Returns
        -------
        cupy.ndarray
            Inference input.

        Raises
        ------
        KeyError
            When no matching input tensor exists.
        """
        return self.get_tensor(name)


@dataclasses.dataclass
class MultiInferenceNLPMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceNLPMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for NLP workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["input_ids", "input_mask", "seq_ids"]

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count)

    @property
    def input_ids(self):
        """
        Returns token-ids for each string padded with 0s to max_length.

        Returns
        -------
        cupy.ndarray
            The token-ids for each string padded with 0s to max_length.

        """

        return self._get_tensor_prop("input_ids")

    @property
    def input_mask(self):
        """
        Returns mask for token-ids result where corresponding positions identify valid token-id values.

        Returns
        -------
        cupy.ndarray
            The mask for token-ids result where corresponding positions identify valid token-id values.

        """

        return self._get_tensor_prop("input_mask")

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

        return self._get_tensor_prop("seq_ids")


@dataclasses.dataclass
class MultiInferenceFILMessage(MultiInferenceMessage, cpp_class=_messages.MultiInferenceFILMessage):
    """
    A stronger typed version of `MultiInferenceMessage` that is used for FIL workloads. Helps ensure the
    proper inputs are set and eases debugging.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["input__0", "seq_ids"]

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count)

    @property
    def input__0(self):
        """
        Input to FIL model inference.

        Returns
        -------
        cupy.ndarray
            Input data.

        """

        return self._get_tensor_prop("input__0")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            Sequence ids.

        """

        return self._get_tensor_prop("seq_ids")
