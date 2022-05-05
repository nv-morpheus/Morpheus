import dataclasses
import typing

import cupy as cp

import morpheus._lib.messages as neom
from morpheus.messages.data_class_prop import DataClassProp
from morpheus.messages.message_base import MessageData
from morpheus.messages.multi_message import MultiMessage


def get_output(instance: "ResponseMemory", name: str):
    """
    Getter function used with DataClassProp for getting inference output from message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding outputs.
    name : str
        Key used to do lookup in outputs dict of message container.

    Returns
    -------
    cupy.ndarray
        Outputs corresponding to name.

    Raises
    ------
    AttributeError
        If output name does not exist in message container.

    """

    if (name not in instance.outputs):
        raise AttributeError

    return instance.outputs[name]


def set_output(instance: "ResponseMemory", name: str, value):
    """
    Setter function used with DataClassProp for setting output in message containers derived
    from ResponseMemory.

    Parameters
    ----------
    instance : `ResponseMemory`
        Message container holding outputs.
    name : str
        Key used to do lookup in outputs dict of message container.
    value : cupy.ndarray
        Value to set for input.
    """

    # Ensure that we have 2D array here (`ensure_2d` inserts the wrong axis)
    instance.outputs[name] = value if value.ndim == 2 else cp.reshape(value, (value.shape[0], -1))


@dataclasses.dataclass
class ResponseMemory(MessageData, cpp_class=neom.ResponseMemory):
    """
    Output memory block holding the results of inference.
    """
    count: int

    outputs: typing.Dict[str, cp.ndarray] = dataclasses.field(default_factory=dict, init=False)

    def get_output(self, name: str):
        if (name not in self.outputs):
            raise KeyError

        return self.outputs[name]


@dataclasses.dataclass
class ResponseMemoryProbs(ResponseMemory, cpp_class=neom.ResponseMemoryProbs):
    probs: dataclasses.InitVar[cp.ndarray] = DataClassProp(get_output, set_output)

    def __post_init__(self, probs):
        self.probs = probs


@dataclasses.dataclass
class ResponseMemoryAE(ResponseMemoryProbs, cpp_class=None):
    user_id: str = ""


@dataclasses.dataclass
class MultiResponseMessage(MultiMessage, cpp_class=neom.MultiResponseMessage):
    """
    This class contains several inference responses as well as the cooresponding message metadata.

    Parameters
    ----------
    memory : `ResponseMemory`
        This is a response container instance for triton inference requests.
    offset : int
        Offset of each response message into the `ResponseMemory` block.
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
        cupy.ndarray
            Inference outputs.

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
        cupy.ndarray
            Inference output.

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
class MultiResponseProbsMessage(MultiResponseMessage, cpp_class=neom.MultiResponseProbsMessage):
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

    user_id: str
