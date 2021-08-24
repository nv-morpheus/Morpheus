# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import typing

import cupy as cp
import pandas as pd


@dataclasses.dataclass
class MessageData:
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


@dataclasses.dataclass
class MessageMeta:
    """
    This is a container class to hold batch deserialized messages metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows in dataframe.
    input_json : typing.List[str]
        Deserialized input messages.

    """

    df: pd.DataFrame
    input_json: typing.List[str]

    @property
    def count(self) -> int:
        """
        Returns the number of messages in the batch.

        Returns
        -------
        int
            number of messages in the MessageMeta.df

        """

        return len(self.df)


@dataclasses.dataclass
class MultiMessage(MessageData):
    """
    This class holds data for multiple messages at a time. To avoid copying data for slicing operations, it
    holds a reference to a batched metadata object and stores the offset and count into that batch.

    Parameters
    ----------
    meta : morpheus.messages.MessageMeta
        Deserialized messages metadata for large batch
    mess_offset : int
        Offset into the metadata batch
    mess_count : int
        Messages count

    """

    meta: MessageMeta = dataclasses.field(repr=False)
    mess_offset: int
    mess_count: int

    @property
    def input_json(self):
        """
        Returns deserialized input rows as a list.

        Returns
        -------
        List[str]
            Input rows

        """

        return self.meta.input_json[self.mess_offset:self.mess_offset + self.mess_count]

    @property
    def id_col(self):
        """
        Returns ID column values from morpheus.messages.MessageMeta.df

        Returns
        -------
        pandas.Series()
            ID column values from the dataframe

        """
        return self.get_meta("ID")

    @property
    def id(self) -> typing.List[int]:
        """
        Returns ID column values from morpheus.messages.MessageMeta.df as list.

        Returns
        -------
        List[int]
            ID column values from the dataframe as list

        """

        return self.get_meta_list("ID")

    @property
    def timestamp(self) -> typing.List[int]:
        """
        Returns timestamp column values from morpheus.messages.MessageMeta.df as list.

        Returns
        -------
        List[int]
            Timestamp column values from the dataframe as list

        """

        return self.get_meta_list("timestamp")

    def get_meta(self, col_name: str):
        """
        Return a column values from morpheus.messages.MessageMeta.df

        Parameters
        ----------
        col_name : str
            Column name in the dataframe.

        Returns
        -------
        pandas.Series()
            Column values from the dataframe.

        """

        return self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], col_name]

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

        return self.get_meta(col_name=col_name).to_list()

    def set_meta(self, col_name: str, value):
        """
        Set column values to morpheus.messages.MessageMeta.df

        Parameters
        ----------
        col_name : str
            Column name in the dataframe.
        value : List
            Column values.

        """

        self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], col_name] = value


@dataclasses.dataclass
class InferenceMemory(MessageData):
    """
    This is a base container class for data that will be used for inference stages. This class is designed to
    hold generic tensor data in cupy arrays.

    Parameters
    ----------
    count : int
        Number of inference inputs
    inputs : typing.Dict[str, cp.ndarray]
        Inference inputs to model.

    """

    count: int

    inputs: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)


class DataClassProp:
    def __init__(self,
                 fget: typing.Callable[[typing.Any, str], typing.Any] = None,
                 fset: typing.Callable[[typing.Any, str, typing.Any], None] = None,
                 fdel=None,
                 doc=None,
                 field=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
        self._field = field

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if (instance is None):
            # Most likely, this is getting the default field value for the dataclass.
            return self._field

        if self.fget is None:
            raise AttributeError("unreadable attribute")

        return self.fget(instance, self.name)

    def __set__(self, instance, value):

        if (instance is None):
            return

        if self.fset is None:
            raise AttributeError("can't set attribute")

        self.fset(instance, self.name, value)

    def __delete__(self, instance):
        if (instance is None):
            return

        del instance.inputs[self.name]


def get_input(instance, name: str):
    if (name not in instance.inputs):
        raise AttributeError

    return instance.inputs[name]


def set_input(instance, name: str, value):
    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.inputs[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass
class InferenceMemoryNLP(InferenceMemory):
    """
    This is a container class for data that needs to be submitted to the inference server for NLP category
    usecases.

    Parameters
    ----------
    input_ids : cp.ndarray
        The token-ids for each string padded with 0s to max_length.
    input_mask : cp.ndarray
        The mask for token-ids result where corresponding positions identify valid token-id values.
    seq_ids : cp.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e. If some messages get broken into multiple inference requests)

    """

    input_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    input_mask: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __post_init__(self, input_ids, input_mask, seq_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.seq_ids = seq_ids


@dataclasses.dataclass
class InferenceMemoryFIL(InferenceMemory):
    """
    This is a container class for data that needs to be submitted to the inference server for FIL category
    usecases.

    Parameters
    ----------
    input__0 : cp.ndarray
        Inference input.
    seq_ids : cp.ndarray
        Ids used to index from an inference input to a message. Necessary since there can be more inference
        inputs than messages (i.e. If some messages get broken into multiple inference requests)

    """

    input__0: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)
    seq_ids: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_input, set_input)

    def __post_init__(self, input__0, seq_ids):
        self.input__0 = input__0
        self.seq_ids = seq_ids


@dataclasses.dataclass
class MultiInferenceMessage(MultiMessage):
    """
    This is a container class that holds the InferenceMemory container and the metadata of the data contained
    within it. Builds on top of the `MultiMessage` class to add additional data for inferencing.

    This class requires two separate memory blocks for a batch. One for the message metadata (i.e. start time,
    IP address, etc.) and another for the raw inference inputs (i.e. input_ids, seq_ids). Since there can be
    more inference input requests than messages (This happens when some messages get broken into multiple
    inference requests) this class stores two different offset and count values. `mess_offset` and
    `mess_count` refer to the offset and count in the message metadata batch and `offset` and `count` index
    into the inference batch data.

    Parameters
    ----------
    memory : morpheus.messages.InferenceMemory
        Inference memory
    offset : int
        Message offset in inference memory instance.
    count : int
        Message count in inference memory instance.

    """

    memory: InferenceMemory = dataclasses.field(repr=False)
    offset: int
    count: int

    @property
    def inputs(self):
        """
        Get inputs stored in the InferenceMemory container.

        Returns
        -------
        cp.ndarray
            Inference inputs

        """

        return {key: self.get_input(key) for key in self.memory.inputs.keys()}

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d

    def __getattr__(self, name: str) -> typing.Any:

        input_val = self.memory.inputs.get(name, None)

        if (input_val is not None):
            return input_val[self.offset:self.offset + self.count, :]

        raise AttributeError

    def get_input(self, name: str):
        """
        Get input stored in the InferenceMemory container.

        Parameters
        ----------
        name : str
            Input key name.

        Returns
        -------
        cp.ndarray
            Inference input

        """

        return self.memory.inputs[name][self.offset:self.offset + self.count, :]

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
        morpheus.messages.MultiInferenceMessage
            A new `MultiInferenceMessage` with sliced offset and count.

        """
        mess_start = self.seq_ids[start, 0].item()
        mess_stop = self.seq_ids[stop - 1, 0].item() + 1
        return MultiInferenceMessage(meta=self.meta,
                                     mess_offset=mess_start,
                                     mess_count=mess_stop - mess_start,
                                     memory=self.memory,
                                     offset=start,
                                     count=stop - start)


@dataclasses.dataclass
class MultiInferenceNLPMessage(MultiInferenceMessage):
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
            inference inputs than messages (i.e. If some messages get broken into multiple inference requests)

        """

        return self.get_input("seq_ids")


@dataclasses.dataclass
class MultiInferenceFILMessage(MultiInferenceMessage):
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
            Input data

        """

        return self.get_input("input__0")

    @property
    def seq_ids(self):
        """
        Returns sequence ids, which are used to keep track of messages in a multi-threaded environment.

        Returns
        -------
        cupy.ndarray
            seq_ids

        """

        return self.get_input("seq_ids")


def get_output(instance: "ResponseMemory", name: str):
    if (name not in instance.outputs):
        raise AttributeError

    return instance.outputs[name]


def set_output(instance: "ResponseMemory", name: str, value):
    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.outputs[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass
class ResponseMemory(MessageData):
    """
    Output memory block holding the results of inference.
    """
    count: int

    outputs: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)


@dataclasses.dataclass
class ResponseMemoryProbs(ResponseMemory):

    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)

    def __post_init__(self, probs):
        self.probs = probs


@dataclasses.dataclass
class MultiResponseMessage(MultiMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : morpheus.messages.ResponseMemory
        This is a response container instance for triton inference requests.
    offset : int
        Offset of each response message into the `ResponseMemory` block
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
        cp.ndarray
            Inference outputs

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
        cp.ndarray
            Inference output

        """

        return self.memory.outputs[name][self.offset:self.offset + self.count, :]

    # def get_slice(self, start, stop):
    #     """
    #     Returns sliced batches based on offsets supplied. Automatically calculates the correct `mess_offset`
    #     and `mess_count`.

    #     Parameters
    #     ----------
    #     start : int
    #         Start offset address.
    #     stop : int
    #         Stop offset address.

    #     Returns
    #     -------
    #     morpheus.messages.MultiResponseMessage
    #         A new `MultiResponseMessage` with sliced offset and count.

    #     """
    #     mess_start = self.seq_ids[start, 0].item()
    #     mess_stop = self.seq_ids[stop - 1, 0].item() + 1
    #     return MultiResponseMessage(meta=self.meta,
    #                                  mess_offset=mess_start,
    #                                  mess_count=mess_stop - mess_start,
    #                                  memory=self.memory,
    #                                  offset=start,
    #                                  count=stop - start)


@dataclasses.dataclass
class MultiResponseProbsMessage(MultiResponseMessage):
    """
    A stronger typed version of `MultiResponseMessage` that is used for inference workloads that return a probability
    array. Helps ensure the proper outputs are set and eases debugging.
    """
    @property
    def probs(self):
        """
        Probabilities of prediction

        Returns
        -------
        cp.ndarray
            probs

        """

        return self.get_output("probs")
