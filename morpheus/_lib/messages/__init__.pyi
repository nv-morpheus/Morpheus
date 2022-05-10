"""
        -----------------------
        .. currentmodule:: morpheus.messages
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        """
from __future__ import annotations
import morpheus._lib.messages
import typing
import cupy
import morpheus._lib.common
import neo.core.node

__all__ = [
    "DataTable",
    "InferenceMemory",
    "InferenceMemoryFIL",
    "InferenceMemoryNLP",
    "MessageMeta",
    "MultiInferenceFILMessage",
    "MultiInferenceMessage",
    "MultiInferenceNLPMessage",
    "MultiMessage",
    "MultiResponseMessage",
    "MultiResponseProbsMessage",
    "ResponseMemory",
    "ResponseMemoryProbs",
    "cupy"
]


class DataTable():
    pass
class InferenceMemory():
    @property
    def count(self) -> int:
        """
        :type: int
        """
    pass
class InferenceMemoryFIL(InferenceMemory):
    def __init__(self, count: int, input__0: object, seq_ids: object) -> None: ...
    def get_tensor(self, arg0: str) -> morpheus._lib.common.Tensor: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def input__0(self) -> object:
        """
        :type: object
        """
    @input__0.setter
    def input__0(self, arg1: object) -> None:
        pass
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    @seq_ids.setter
    def seq_ids(self, arg1: object) -> None:
        pass
    pass
class InferenceMemoryNLP(InferenceMemory):
    def __init__(self, count: int, input_ids: object, input_mask: object, seq_ids: object) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def input_ids(self) -> object:
        """
        :type: object
        """
    @input_ids.setter
    def input_ids(self, arg1: object) -> None:
        pass
    @property
    def input_mask(self) -> object:
        """
        :type: object
        """
    @input_mask.setter
    def input_mask(self, arg1: object) -> None:
        pass
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    @seq_ids.setter
    def seq_ids(self, arg1: object) -> None:
        pass
    pass
class MessageMeta():
    def __init__(self, df: object) -> None: ...
    @staticmethod
    def make_from_file(arg0: str) -> MessageMeta: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def df(self) -> object:
        """
        :type: object
        """
    pass
class MultiMessage():
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int) -> None: ...
    @typing.overload
    def get_meta(self) -> object: ...
    @typing.overload
    def get_meta(self, arg0: str) -> object: ...
    @typing.overload
    def get_meta(self, arg0: typing.List[str]) -> object: ...
    def get_slice(self, arg0: int, arg1: int) -> MultiMessage: ...
    def set_meta(self, arg0: object, arg1: object) -> None: ...
    @property
    def mess_count(self) -> int:
        """
        :type: int
        """
    @property
    def mess_offset(self) -> int:
        """
        :type: int
        """
    @property
    def meta(self) -> MessageMeta:
        """
        :type: MessageMeta
        """
    pass
class MultiInferenceMessage(MultiMessage):
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int, memory: InferenceMemory, offset: int, count: int) -> None: ...
    def get_input(self, arg0: str) -> object: ...
    def get_slice(self, arg0: int, arg1: int) -> MultiInferenceMessage: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def memory(self) -> InferenceMemory:
        """
        :type: InferenceMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    pass
class MultiInferenceNLPMessage(MultiInferenceMessage, MultiMessage):
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int, memory: InferenceMemory, offset: int, count: int) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def input_ids(self) -> object:
        """
        :type: object
        """
    @property
    def input_mask(self) -> object:
        """
        :type: object
        """
    @property
    def memory(self) -> InferenceMemory:
        """
        :type: InferenceMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    @property
    def seq_ids(self) -> object:
        """
        :type: object
        """
    pass
class MultiInferenceFILMessage(MultiInferenceMessage, MultiMessage):
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int, memory: InferenceMemory, offset: int, count: int) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def memory(self) -> InferenceMemory:
        """
        :type: InferenceMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    pass
class MultiResponseMessage(MultiMessage):
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int, memory: ResponseMemory, offset: int, count: int) -> None: ...
    def get_output(self, arg0: str) -> object: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def memory(self) -> ResponseMemory:
        """
        :type: ResponseMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    pass
class MultiResponseProbsMessage(MultiResponseMessage, MultiMessage):
    def __init__(self, meta: MessageMeta, mess_offset: int, mess_count: int, memory: ResponseMemory, offset: int, count: int) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def memory(self) -> ResponseMemory:
        """
        :type: ResponseMemory
        """
    @property
    def offset(self) -> int:
        """
        :type: int
        """
    @property
    def probs(self) -> object:
        """
        :type: object
        """
    pass
class ResponseMemory():
    def get_output(self, arg0: str) -> object: ...
    def get_output_tensor(self, arg0: str) -> morpheus._lib.common.Tensor: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    pass
class ResponseMemoryProbs(ResponseMemory):
    def __init__(self, count: int, probs: object) -> None: ...
    @property
    def count(self) -> int:
        """
        :type: int
        """
    @property
    def probs(self) -> object:
        """
        :type: object
        """
    @probs.setter
    def probs(self, arg1: object) -> None:
        pass
    pass
__version__ = 'dev'
