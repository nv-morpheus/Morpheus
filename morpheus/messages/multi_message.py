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
import numpy as np

import cudf

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

    """
    meta: MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int

    def __init__(self, meta: MessageMeta, mess_offset: int = 0, mess_count: int = -1):

        if meta is None:
            raise ValueError("Must define `meta` when creating MultiMessage")

        # Use the meta count if not supplied
        if (mess_count == -1):
            mess_count = meta.count

        # Check for valid offsets and counts
        if mess_offset < 0 or mess_offset >= meta.count:
            raise ValueError("Invalid message offset value")
        if mess_count <= 0 or (mess_offset + mess_count > meta.count):
            raise ValueError("Invalid message count value")

        self.meta = meta
        self.mess_offset = mess_offset
        self.mess_count = mess_count

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

    def _get_indexers(self, columns: typing.Union[None, str, typing.List[str]] = None):
        row_indexer = slice(self.mess_offset, self.mess_offset + self.mess_count, 1)

        if (columns is None):
            columns = self.meta._df.columns.to_list()
        elif (isinstance(columns, str)):
            # Convert a single string into a list so all versions return tables, not series
            columns = [columns]

        column_indexer = self.meta._df.columns.get_indexer_for(columns)

        return row_indexer, column_indexer

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

        row_indexer, column_indexer = self._get_indexers(columns=columns)

        if (-1 in column_indexer):
            missing_columns = [columns[i] for i, index_value in enumerate(column_indexer) if index_value == -1]
            raise KeyError("Requested columns {} does not exist in the dataframe".format(missing_columns))
        elif (isinstance(columns, str) and len(column_indexer) == 1):
            # Make sure to return a series for a single column
            column_indexer = column_indexer[0]

        return self.meta._df.iloc[row_indexer, column_indexer]

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

        # First try to set the values on just our slice if the columns exist
        row_indexer, column_indexer = self._get_indexers(columns=columns)

        # Get exclusive access to the dataframe
        with self.meta.mutable_dataframe() as df:
            # Check to see if we are adding a column. If so, we need to use df.loc instead of df.iloc
            if (-1 not in column_indexer):

                # If we only have one column, convert it to a series (broadcasts work with more types on a series)
                if (len(column_indexer) == 1):
                    column_indexer = column_indexer[0]

                # Now update the slice
                df.iloc[row_indexer, column_indexer] = value

            else:
                # Need to determine the boolean mask to use indexes with df.loc
                row_mask = self._ranges_to_mask(self.meta._df, [(self.mess_offset, self.mess_offset + self.mess_count)])

                # Columns should never be empty if we get here
                assert columns is not None

                # Now set the slice
                df.loc[row_mask, columns] = value

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

        offset = self.mess_offset + start
        count = stop - start

        if (start < 0 or start >= self.mess_count):
            raise IndexError("Invalid `start` argument")
        if (stop <= start or count > self.mess_count):
            raise IndexError("Invalid `stop` argument")

        return MultiMessage(meta=self.meta, mess_offset=offset, mess_count=count)

    def _ranges_to_mask(self, df, ranges):
        if isinstance(df, cudf.DataFrame):
            zeros_fn = cp.zeros
        else:
            zeros_fn = np.zeros

        mask = zeros_fn(len(df), bool)

        for range in ranges:
            mask[range[0]:range[1]] = True

        return mask

    def copy_meta_ranges(self,
                         ranges: typing.List[typing.Tuple[int, int]],
                         mask: typing.Union[None, cp.ndarray, np.ndarray] = None):
        """
        Perform a copy of the underlying dataframe for the given `ranges` of rows.

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
        `Dataframe`
        """
        df = self.get_meta()

        if mask is None:
            mask = self._ranges_to_mask(df, ranges=ranges)

        return df.loc[mask, :]

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message instance for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        Returns
        -------
        `MultiMessage`
        """
        sliced_rows = self.copy_meta_ranges(ranges)
        return MultiMessage(meta=MessageMeta(sliced_rows), mess_offset=0, mess_count=len(sliced_rows))
