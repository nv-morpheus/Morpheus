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
from morpheus.messages.memory.response_memory import ResponseMemory
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.multi_tensor_message import MultiTensorMessage


@dataclasses.dataclass
class MultiResponseMessage(MultiTensorMessage, cpp_class=_messages.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.
    """

    @property
    def outputs(self):
        """
        Get outputs stored in the ResponseMemory container. Alias for `MultiResponseMessage.tensors`.

        Returns
        -------
        cupy.ndarray
            Inference outputs.

        """
        return self.tensors

    def get_output(self, name: str):
        """
        Get output stored in the ResponseMemory container. Alias for `MultiResponseMessage.get_tensor`.

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

    def copy_output_ranges(self, ranges, mask=None):
        """
        Perform a copy of the underlying output tensors for the given `ranges` of rows.
        Alias for `MultiResponseMessage.copy_output_ranges`

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
        return self.copy_tensor_ranges(ranges, mask=mask)

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message, dataframe and tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        -------
        `MultiResponseMessage`
        """
        m = super().copy_ranges(ranges)
        return MultiResponseMessage(meta=m.meta,
                                    mess_offset=m.mess_offset,
                                    mess_count=m.mess_count,
                                    memory=m.memory,
                                    offset=m.offset,
                                    count=m.mess_count)

    def get_slice(self, start, stop):
        """
        Perform a slice of the current message from `start`:`stop` (excluding `stop`)

        For example to slice from rows 1-3 use `m.get_slice(1, 4)`. The returned `MultiResponseMessage` will contain
        references to the same underlying Dataframe and output tensors, and this calling this method is reletively low
        cost compared to `MultiResponseMessage.copy_ranges`

        Parameters
        ----------
        start : int
            Starting row of the slice

        stop : int
            Stop of the slice

        -------
        `MultiResponseMessage`
        """
        mess_count = stop - start
        return MultiResponseMessage(meta=self.meta,
                                    mess_offset=self.mess_offset + start,
                                    mess_count=mess_count,
                                    memory=self.memory,
                                    offset=self.offset + start,
                                    count=mess_count)


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

    user_id: str = None

    def copy_ranges(self, ranges: typing.List[typing.Tuple[int, int]]):
        """
        Perform a copy of the current message, dataframe and tensors for the given `ranges` of rows.

        Parameters
        ----------
        ranges : typing.List[typing.Tuple[int, int]]
            Rows to include in the copy in the form of `[(`start_row`, `stop_row`),...]`
            The `stop_row` isn't included. For example to copy rows 1-2 & 5-7 `ranges=[(1, 3), (5, 8)]`

        -------
        `MultiResponseAEMessage`
        """
        m = super().copy_ranges(ranges)
        return MultiResponseAEMessage(meta=m.meta,
                                      mess_offset=m.mess_offset,
                                      mess_count=m.mess_count,
                                      memory=m.memory,
                                      offset=m.offset,
                                      count=m.mess_count,
                                      user_id=self.user_id)

    def get_slice(self, start, stop):
        """
        Perform a slice of the current message from `start`:`stop` (excluding `stop`)

        For example to slice from rows 1-3 use `m.get_slice(1, 4)`. The returned `MultiResponseMessage` will contain
        references to the same underlying Dataframe and output tensors, and this calling this method is reletively low
        cost compared to `MultiResponseAEMessage.copy_ranges`

        Parameters
        ----------
        start : int
            Starting row of the slice

        stop : int
            Stop of the slice

        -------
        `MultiResponseAEMessage`
        """
        slice = super().get_slice(start, stop)
        return MultiResponseAEMessage(meta=slice.meta,
                                      mess_offset=slice.mess_offset,
                                      mess_count=slice.mess_count,
                                      memory=slice.memory,
                                      offset=slice.offset,
                                      count=slice.mess_count,
                                      user_id=self.user_id)
