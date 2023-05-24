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
from morpheus.utils import logger as morpheus_logger

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MultiResponseMessage(_messages.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.
    """

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

        # # Add the tensor name to the required list
        # if (self.probs_tensor_name not in self.required_tensors):
        #     # Make sure to set a new variable here instead of append otherwise you change all classes
        #     self.required_tensors = self.required_tensors + [self.probs_tensor_name]

        super().__init__(meta=meta,
                         mess_offset=mess_offset,
                         mess_count=mess_count,
                         memory=memory,
                         offset=offset,
                         count=count,
                         id_tensor_name=id_tensor_name,
                         probs_tensor_name=probs_tensor_name)

    Self = typing.TypeVar("Self", bound="MultiResponseMessage")

    @classmethod
    def from_message(cls: typing.Type[Self],
                     message: "MultiResponseMessage",
                     *,
                     meta: MessageMeta = None,
                     mess_offset: int = -1,
                     mess_count: int = -1,
                     **kwargs) -> Self:

        import inspect
        """
        Creates a new instance of a derived class from `MultiMessage` using an existing message as the template. This is
        very useful when a new message needs to be created with a single change to an existing `MessageMeta`.

        When creating the new message, all required arguments for the class specified by `cls` will be pulled from
        `message` unless otherwise specified in the `args` or `kwargs`. Special handling is performed depending on
        whether or not a new `meta` object is supplied. If one is supplied, the offset and count defaults will be 0 and
        `meta.count` respectively. Otherwise offset and count will be pulled from the input `message`.

        Parameters
        ----------
        cls : typing.Type[Self]
            The class to create
        message : MultiMessage
            An existing message to use as a template. Can be a base or derived from `cls` as long as all arguments can
            be pulled from `message` or proveded in `kwargs`
        meta : MessageMeta, optional
            A new `MessageMeta` to use, by default None
        mess_offset : int, optional
            A new `mess_offset` to use, by default -1
        mess_count : int, optional
            A new `mess_count` to use, by default -1

        Returns
        -------
        Self
            A new instance of type `cls`

        Raises
        ------
        ValueError
            If the incoming `message` is None
        AttributeError
            If some required arguments were not supplied by `kwargs` and could not be pulled from `message`
        """

        if (message is None):
            raise ValueError("Must define `message` when creating a MultiMessage with `from_message`")

        if (mess_offset == -1):
            if (meta is not None):
                mess_offset = 0
            else:
                mess_offset = message.mess_offset

        if (mess_count == -1):
            if (meta is not None):
                # Subtract offset here so we dont go over the end
                mess_count = meta.count - mess_offset
            else:
                mess_count = message.mess_count

        # Do meta last
        if meta is None:
            meta = message.meta

        # Update the kwargs
        kwargs.update({
            "meta": meta,
            "mess_offset": mess_offset,
            "mess_count": mess_count,
        })

        signature = inspect.signature(cls.__init__)

        for p_name, param in signature.parameters.items():

            if (p_name == "self"):
                # Skip self until this is fixed (python 3.9) https://github.com/python/cpython/issues/85074
                # After that, switch to using inspect.signature(cls)
                continue

            # Skip if its already defined
            if (p_name in kwargs):
                continue

            if (not hasattr(message, p_name)):
                # Check for a default
                if (param.default == inspect.Parameter.empty):
                    raise AttributeError(
                        f"Cannot create message of type {cls}, from {message}. Missing property '{p_name}'")

                # Otherwise, we can ignore
                continue

            kwargs[p_name] = getattr(message, p_name)

        # Create a new instance using the kwargs
        return cls(**kwargs)


@dataclasses.dataclass
class MultiResponseProbsMessage(MultiResponseMessage): # , cpp_class=_messages.MultiResponseProbsMessage
    """
    A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
    array. Helps ensure the proper outputs are set and eases debugging.
    """

    required_tensors: typing.ClassVar[typing.List[str]] = ["probs"]

    def __new__(cls, *args, **kwargs):
        morpheus_logger.deprecated_message_warning(logger, cls, MultiResponseMessage)
        return super(MultiResponseMessage, cls).__new__(cls)

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
class MultiResponseAEMessage(MultiResponseMessage): # , cpp_class=None
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
