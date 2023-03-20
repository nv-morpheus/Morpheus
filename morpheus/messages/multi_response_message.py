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
import logging
import typing

import morpheus._lib.messages as _messages
from morpheus.messages.memory.tensor_memory import TensorMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_tensor_message import MultiTensorMessage
from morpheus.utils import logger as morpheus_logger

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiResponseMessage(MultiTensorMessage, cpp_class=_messages.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.
    """

    probs_tensor_name: typing.ClassVar[str] = "probs"
    """Name of the tensor that holds output probabilities"""

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1,
                 id_tensor_name: str = "seq_ids",
                 probs_tensor_name: str = "probs"):

        if probs_tensor_name is None:
            raise ValueError("Cannot use None for `probs_tensor_name`")

        self.probs_tensor_name = probs_tensor_name

        # Add the tensor name to the required list
        if (self.probs_tensor_name not in self.required_tensors):
            # Make sure to set a new variable here instead of append otherwise you change all classes
            self.required_tensors = self.required_tensors + [self.probs_tensor_name]

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count,
                         id_tensor_name=id_tensor_name)

    @property
    def outputs(self):
        """
        Get outputs stored in the TensorMemory container. Alias for `MultiResponseMessage.tensors`.

        Returns
        -------
        cupy.ndarray
            Inference outputs.

        """
        return self.tensors

    def get_output(self, name: str):
        """
        Get output stored in the TensorMemory container. Alias for `MultiResponseMessage.get_tensor`.

        Parameters
        ----------
        name : str
            Output key name.

        Returns
        -------
        cupy.ndarray
            Inference output.

        """
        return self.get_tensor(name)

    def get_probs_tensor(self):
        """
        Get the tensor that holds output probabilities. Equivalent to `get_tensor(probs_tensor_name)`

        Returns
        -------
        cupy.ndarray
            The probabilities tensor
        """

        return self.get_tensor(self.probs_tensor_name)


@dataclasses.dataclass
class MultiResponseProbsMessage(MultiResponseMessage, cpp_class=_messages.MultiResponseProbsMessage):
    """
    A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
    array. Helps ensure the proper outputs are set and eases debugging.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["probs"]

    def __new__(cls, *args, **kwargs):
        morpheus_logger.deprecated_message_warning(logger, cls, MultiResponseMessage)
        return super(MultiResponseMessage, cls).__new__(cls, *args, **kwargs)

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory,
                 offset: int = 0,
                 count: int = -1,
                 id_tensor_name: str = "seq_ids",
                 probs_tensor_name: str = "probs"):

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count,
                         id_tensor_name=id_tensor_name,
                         probs_tensor_name=probs_tensor_name)

    @property
    def probs(self):
        """
        Probabilities of prediction.

        Returns
        -------
        cupy.ndarray
            probabilities

        """

        return self._get_tensor_prop("probs")


@dataclasses.dataclass
class MultiResponseAEMessage(MultiResponseMessage, cpp_class=None):
    """
    A stronger typed version of `MultiResponseProbsMessage` that is used for inference workloads that return a
    probability array. Helps ensure the proper outputs are set and eases debugging.
    """

    user_id: str = None

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory = None,
                 offset: int = 0,
                 count: int = -1,
                 id_tensor_name: str = "seq_ids",
                 probs_tensor_name: str = "probs",
                 user_id: str = None):

        if (user_id is None):
            raise ValueError("Must define `user_id` when creating {}".format(self.__class__.__name__))

        self.user_id = user_id

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count,
                         id_tensor_name=id_tensor_name,
                         probs_tensor_name=probs_tensor_name)
