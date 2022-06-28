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

import cupy

import morpheus._lib.messages as _messages
from morpheus.messages.message_base import MessageData
from morpheus.messages.message_meta import MessageMeta


@dataclasses.dataclass
class MultiMessage(MessageData, cpp_class=_messages.MultiMessage):
    """
    This class holds data for multiple messages at a time. To avoid copying data for slicing operations, it
    holds a reference to a batched metadata object and stores the offset and count into that batch.

    Parameters
    ----------
    meta : `MessageMeta`
        Deserialized messages metadata for large batch.
    mess_offset : int
        Offset into the metadata batch.
    mess_count : int
        Messages count.
    mask : cupy.array
        Optional boolean array.
        If `None` this will be computed from the values specified by `mess_offset` and `mess_count`
        If not-None then `mess_offset` and `mess_count` will be ignored

    """
    meta: MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int
    mask: cupy.array = dataclasses.field(repr=False, init=False)

    def __post_init__(self):
        self.mask = cupy.zeros(self.meta.count, dtype=cupy.bool_)
        self.mask[self.mess_offset:self.mess_offset + self.mess_count] = True

    @property
    def id_col(self):
        """
        Returns ID column values from `morpheus.pipeline.messages.MessageMeta.df`.

        Returns
        -------
        pandas.Series
            ID column values from the dataframe.

        """
        return self.get_meta("ID")

    @property
    def id(self) -> typing.List[int]:
        """
        Returns ID column values from `morpheus.pipeline.messages.MessageMeta.df` as list.

        Returns
        -------
        List[int]
            ID column values from the dataframe as list.

        """

        return self.get_meta_list("ID")

    @property
    def timestamp(self) -> typing.List[int]:
        """
        Returns timestamp column values from morpheus.messages.MessageMeta.df as list.

        Returns
        -------
        List[int]
            Timestamp column values from the dataframe as list.

        """

        return self.get_meta_list("timestamp")

    def get_meta(self, columns: typing.Union[None, str, typing.List[str]] = None):
        """
        Return column values from `morpheus.pipeline.messages.MessageMeta.df`.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Returns all columns if `None` is specified. When a string is passed, a `Series` is
            returned. Otherwise a `Dataframe` is returned.

        Returns
        -------
        Series or Dataframe
            Column values from the dataframe.

        """

        if (columns is None):
            return self.meta.df.loc[self.mask, :]
        else:
            return self.meta.df.loc[self.mask, columns]

    def get_meta_list(self, col_name: str = None):
        """
        Return a column values from morpheus.messages.MessageMeta.df as a list.

        Parameters
        ----------
        col_name : str
            Column name in the dataframe.

        Returns
        -------
        List[str]
            Column values from the dataframe.

        """

        return self.get_meta(col_name).to_arrow().to_pylist()
    
    def set_meta(self, columns: typing.Union[None, str, typing.List[str]], value):
        """
        Set column values to `morpheus.pipelines.messages.MessageMeta.df`.

        Parameters
        ----------
        columns : typing.Union[None, str, typing.List[str]]
            Input column names. Sets the value for the corresponding column names. If `None` is specified, all columns
            will be used. If the column does not exist, a new one will be created.
        value : Any
            Value to apply to the specified columns. If a single value is passed, it will be broadcast to all rows. If a
            `Series` or `Dataframe` is passed, rows will be matched by index.

        """
        if (columns is None):
            # Set all columns
            self.meta.df.loc[self.mask, :] = value
        else:
            # If its a single column or list of columns, this is the same
            self.meta.df.loc[self.mask, columns] = value

    def get_slice(self, start, stop, mask: cupy.array = None):
        """
        Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
        and `mess_count`.

        Parameters
        ----------
        start : int
            Start offset address.
        stop : int
            Stop offset address.
        mask : cupy.array
            Optional boolean array of masked values

        Returns
        -------
        `MultiMessage`
            A new `MultiMessage` with sliced offset and count.

        """
        sliced = MultiMessage(meta=self.meta, mess_offset=start, mess_count=stop - start)
        sliced.mask = self.mask
        return sliced
