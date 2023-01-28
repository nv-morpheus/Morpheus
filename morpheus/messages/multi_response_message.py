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
from morpheus.messages.message_meta import MessageMeta
from morpheus.messages.multi_message import MultiMessage
from morpheus.messages.tensor_memory import TensorMemory


def get_output(instance: "ResponseMemory", name: str):
    """
    Getter function used with DataClassProp for getting inference output from message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding tensors.
    name : str
        Key used to do lookup in tensors dict of message container.

    Returns
    -------
    cupy.ndarray
        Tensors corresponding to name.

    Raises
    ------
    AttributeError
        If output name does not exist in message container.

    """

    if (name not in instance.tensors):
        raise AttributeError

    return instance.tensors[name]


def set_output(instance: "ResponseMemory", name: str, value):
    """
    Setter function used with DataClassProp for setting output in message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding tensors.
    name : str
        Key used to do lookup in tensors dict of message container.
    value : cupy.ndarray
        Value to set for input.
    """

    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.tensors[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass
class ResponseMemory(TensorMemory, cpp_class=_messages.ResponseMemory):
    """Output memory block holding the results of inference."""

    def get_output(self, name: str):
        """
        Return the output tensor specified by `name`.

        Parameters
        ----------
        name : str
            Name of output tensor.

        Returns
        -------
        cupy.ndarray
            Tensor corresponding to name.
        """
        return self.tensors[name]


@dataclasses.dataclass(init=False)
class ResponseMemoryProbs(ResponseMemory, cpp_class=_messages.ResponseMemoryProbs):
    """
    Subclass of `ResponseMemory` containng an output tensor named 'probs'.

    Parameters
    ----------
    probs : cupy.ndarray
        Probabilities tensor
    """
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)

    def __init__(self, probs):
        super().__init__(count=len(probs), tensors={'probs': probs})


@dataclasses.dataclass(init=False)
class ResponseMemoryAE(ResponseMemory, cpp_class=None):
    """
    Subclass of `ResponseMemory` specific to the AutoEncoder pipeline.

    Parameters
    ----------
    probs : cupy.ndarray
        Probabilities tensor

    user_id : str
        User id the inference was performed against.

    explain_df : pd.Dataframe
        Explainability Dataframe, for each feature a column will exist with a name in the form of: `{feature}_z_loss`
        containing the loss z-score along with `max_abs_z` and `mean_abs_z` columns
    """
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)
    user_id = ""
    explain_df = None

    def __init__(self, probs):
        super().__init__(count=len(probs), tensors={'probs': probs})


@dataclasses.dataclass
class MultiResponseMessage(MultiMessage, cpp_class=_messages.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : `TensorMemory`
        This is a response container instance for triton inference requests.
    offset : int
        Offset of each response message into the `TensorMemory` block.
    count : int
        Inference results size of all responses.

    """
    memory: TensorMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def outputs(self):
        """
        Get outputs stored in the TensorMemory container.

        Returns
        -------
        cupy.ndarray
            Inference outputs.

        """

        return {key: self.get_output(key) for key in self.memory.tensors.keys()}

    def __getattr__(self, name: str) -> typing.Any:

        output_val = self.memory.tensors.get(name, None)

        if (output_val is not None):
            return output_val[self.offset:self.offset + self.count, :]

        raise AttributeError

    def get_output(self, name: str):
        """
        Get output stored in the TensorMemory container.

        Parameters
        ----------
        name : str
            Output key name.

        Returns
        -------
        cupy.ndarray
            Inference output.

        """

        return self.memory.tensors[name][self.offset:self.offset + self.count, :]

    def copy_output_ranges(self, ranges, mask=None):
        """
        Perform a copy of the underlying output tensors for the given `ranges` of rows.

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

        # The outputs property method returns a copy with the offsets applied
        outputs = self.outputs
        return {key: output[mask] for (key, output) in outputs.items()}

    def copy_ranges(self, ranges):
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
        mask = self._ranges_to_mask(self.get_meta(), ranges)
        sliced_rows = self.copy_meta_ranges(ranges, mask=mask)
        sliced_count = len(sliced_rows)
        sliced_outputs = self.copy_output_ranges(ranges, mask=mask)

        mem = TensorMemory(count=sliced_count)
        mem.outputs = sliced_outputs

        return MultiResponseMessage(MessageMeta(sliced_rows), 0, sliced_count, mem, 0, sliced_count)

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

    def copy_ranges(self, ranges):
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
