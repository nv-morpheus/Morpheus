# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from morpheus.messages.multi_message import MultiMessage

# Needed to provide the return type of `@classmethod`
Self = typing.TypeVar("Self", bound="MultiTensorMessage")


@dataclasses.dataclass
class MultiTensorMessage(MultiMessage, cpp_class=_messages.MultiTensorMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : `TensorMemory`
        Container holding generic tensor data in cupy arrays
    offset : int
        Offset of each message into the `TensorMemory` block.
    count : int
        Number of rows in the `TensorMemory` block.

    """
    memory: TensorMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    required_tensors: typing.ClassVar[typing.List[str]] = []
    id_tensor: typing.ClassVar[str] = "seq_ids"

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 memory: TensorMemory,
                 offset: int = 0,
                 count: int = -1):

        if memory is None:
            raise ValueError("Must define `memory` when creating {}".format(self.__class__.__name__))

        # Use the meta count if not supplied
        if (count == -1):
            count = memory.count - offset

        # Check for valid offsets and counts
        if offset < 0 or offset >= memory.count:
            raise ValueError("Invalid offset value")
        if count <= 0 or (offset + count > memory.count):
            raise ValueError("Invalid count value")

        self.memory = memory
        self.offset = offset
        self.count = count

        # Call the base class last because the properties need to be initialized first
        super().__init__(meta=meta, mess_offset=mess_offset, mess_count=mess_count)

        if (self.count < self.mess_count):
            raise ValueError("Invalid count value. Must have a count greater than or equal to mess_count")

        # Finally, check for the required tensors class attribute
        if (hasattr(self.__class__, "required_tensors")):
            for tensor_name in self.__class__.required_tensors:
                if (not memory.has_tensor(tensor_name)):
                    raise ValueError((f"`TensorMemory` object must have a '{tensor_name}' "
                                      f"tensor to create `{self.__class__.__name__}`").format(self.__class__.__name__))

    @property
    def tensors(self):
        """
        Get tensors stored in the TensorMemory container sliced according to `offset` and `count`.

        Returns
        -------
        cupy.ndarray
            Inference tensors.

        """
        tensors = self.memory.get_tensors()
        return {key: self.get_tensor(key) for key in tensors.keys()}

    def __getattr__(self, name: str) -> typing.Any:
        if ("memory" in self.__dict__ and self.memory.has_tensor(name)):
            return self._get_tensor_prop(name)

        if hasattr(super(), "__getattr__"):
            return super().__getattr__(name)
        raise AttributeError

    def _calc_message_slice_bounds(self, start: int, stop: int):

        mess_start = start
        mess_stop = stop

        if (self.count != self.mess_count):
            if (hasattr(self.__class__, "id_tensor") and self.__class__.id_tensor is not None):
                id_tensor_name = self.__class__.id_tensor

                if (not self.memory.has_tensor(id_tensor_name)):
                    raise RuntimeError(f"The tensor memory object is missing the required ID tensor '{id_tensor_name}' "
                                       f"this tensor is required to make slices of MultiTensorMessages")

                id_tensor = self.get_tensor(id_tensor_name)

                # Now determine the new mess_start and mess_stop
                mess_start = id_tensor[start, 0].item()
                mess_stop = id_tensor[stop - 1, 0].item() + 1
            else:
                raise RuntimeError("Cannot calculate slice when the tensor count is different than the message count. "
                                   "Must use a derived class which tracks message IDs")

        # Return the base calculation now
        return super()._calc_message_slice_bounds(start=mess_start, stop=mess_stop)

    def _calc_memory_slice_bounds(self, start: int, stop: int):

        # Start must be between [0, mess_count)
        if (start < 0 or start >= self.count):
            raise IndexError("Invalid memory `start` argument")

        # Stop must be between (start, mess_count]
        if (stop <= start or stop > self.count):
            raise IndexError("Invalid memory `stop` argument")

        # Calculate the new offset and count
        offset = self.offset + start
        count = stop - start

        return offset, count

    def get_tensor(self, name: str):
        """
        Get tensor stored in the TensorMemory container.

        Parameters
        ----------
        name : str
            tensor key name.

        Returns
        -------
        cupy.ndarray
            Inference tensor.

        """
        return self.memory.get_tensor(name)[self.offset:self.offset + self.count, :]

    def _get_tensor_prop(self, name: str):
        """
        This method is intended to be used by propery methods in subclasses

        Parameters
        ----------
        name : str
            Tensor key name.

        Returns
        -------
        cupy.ndarray
            Tensor.

        Raises
        ------
        AttributeError
            If tensor name does not exist in the container.
        """
        try:
            return self.get_tensor(name)
        except KeyError:
            raise AttributeError(f'No attribute named "{name}" exists')

    def copy_tensor_ranges(self, ranges, mask=None):
        """
        Perform a copy of the underlying tensor tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        mask : typing.Union[None, cupy.ndarray, numpy.ndarray]
            Optionally specify rows as a cupy array (when using cudf Dataframes) or a numpy array (when using pandas
            Dataframes) of booleans. When not-None `ranges` will be ignored. This is useful as an optimization as this
            avoids needing to generate the mask on it's own.

        Returns
        -------
        typing.Dict[str, cupy.ndarray]
        """
        if mask is None:
            mask = self._ranges_to_mask(self.get_meta(), ranges=ranges)

        # The tensors property method returns a copy with the offsets applied
        tensors = self.tensors
        return {key: tensor[mask] for (key, tensor) in tensors.items()}

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message, dataframe and tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        -------
        `MultiTensorMessage`
        """
        mask = self._ranges_to_mask(self.get_meta(), ranges)
        sliced_rows = self.copy_meta_ranges(ranges, mask=mask)
        sliced_count = len(sliced_rows)
        sliced_tensors = self.copy_tensor_ranges(ranges, mask=mask)

        mem = TensorMemory(count=sliced_count, tensors=sliced_tensors)

        return self.from_message(self,
                                 meta=MessageMeta(sliced_rows),
                                 mess_offset=0,
                                 mess_count=sliced_count,
                                 memory=mem,
                                 offset=0,
                                 count=sliced_count)

    def get_slice(self, start, stop):
        """
        Perform a slice of the current message from `start`:`stop` (excluding `stop`)

        For example to slice from rows 1-3 use `m.get_slice(1, 4)`. The returned `MultiTensorMessage` will contain
        references to the same underlying Dataframe and tensor tensors, and this calling this method is reletively low
        cost compared to `MultiTensorMessage.copy_ranges`

        Parameters
        ----------
        start : int
            Starting row of the slice

        stop : int
            Stop of the slice

        -------
        `MultiTensorMessage`
        """

        # Calc the offset and count. This checks the bounds for us
        mess_offset, mess_count = self._calc_message_slice_bounds(start=start, stop=stop)
        offset, count = self._calc_memory_slice_bounds(start=start, stop=stop)

        kwargs = {
            "meta": self.meta,
            "mess_offset": mess_offset,
            "mess_count": mess_count,
            "memory": self.memory,
            "offset": offset,
            "count": count,
        }

        return self.from_message(self, **kwargs)

    @classmethod
    def from_message(cls: typing.Type[Self],
                     message: "MultiTensorMessage",
                     *,
                     meta: MessageMeta = None,
                     mess_offset: int = -1,
                     mess_count: int = -1,
                     memory: TensorMemory = None,
                     offset: int = -1,
                     count: int = -1,
                     **kwargs) -> Self:
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
        memory : TensorMemory, optional
            A new `TensorMemory` to use. If supplied, `offset` and `count` default to `0` and `memory.count`
            respectively. By default None
        offset : int, optional
            A new `offset` to use, by default -1
        count : int, optional
            A new `count` to use, by default -1

        Returns
        -------
        Self
            A new instance of type `cls`

        Raises
        ------
        ValueError
            If the incoming `message` is None
        """

        if (message is None):
            raise ValueError("Must define `message` when creating a MultiMessage with `from_message`")

        if (offset == -1):
            if (memory is not None):
                offset = 0
            else:
                offset = message.offset

        if (count == -1):
            if (memory is not None):
                # Subtract offset here so we dont go over the end
                count = memory.count - offset
            else:
                count = message.count

        # Do meta last
        if memory is None:
            memory = message.memory

        # Update the kwargs
        kwargs.update({
            "meta": meta,
            "mess_offset": mess_offset,
            "mess_count": mess_count,
            "memory": memory,
            "offset": offset,
            "count": count,
        })

        return super().from_message(message, **kwargs)
