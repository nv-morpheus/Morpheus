"""
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.llm
import typing
import morpheus._lib.messages
import morpheus._lib.pycoro

__all__ = [
    "InputMap",
    "LLMContext",
    "LLMEngine",
    "LLMNode",
    "LLMNodeBase",
    "LLMNodeRunner",
    "LLMTask",
    "LLMTaskHandler"
]


class InputMap():
    def __init__(self) -> None: ...
    @property
    def input_name(self) -> str:
        """
        :type: str
        """
    @input_name.setter
    def input_name(self, arg0: str) -> None:
        pass
    @property
    def node_name(self) -> str:
        """
        :type: str
        """
    @node_name.setter
    def node_name(self, arg0: str) -> None:
        pass
    pass
class LLMContext():
    @typing.overload
    def get_input(self) -> object: ...
    @typing.overload
    def get_input(self, arg0: str) -> object: ...
    def get_inputs(self) -> dict: ...
    def message(self) -> morpheus._lib.messages.ControlMessage: ...
    def set_output(self, arg0: object) -> None: ...
    def task(self) -> LLMTask: ...
    @property
    def all_outputs(self) -> object:
        """
        :type: object
        """
    @property
    def full_name(self) -> str:
        """
        :type: str
        """
    @property
    def input_map(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def parent(self) -> LLMContext:
        """
        :type: LLMContext
        """
    pass
class LLMNodeBase():
    def __init__(self) -> None: ...
    def execute(self, arg0: LLMContext) -> typing.Awaitable[LLMContext]: ...
    pass
class LLMNode(LLMNodeBase):
    def __init__(self) -> None: ...
    @typing.overload
    def add_node(self, name: str, *, inputs: typing.List[typing.Union[str, typing.Tuple[str, str]]], node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner: ...
    @typing.overload
    def add_node(self, name: str, *, node: LLMNodeBase, is_output: bool = False) -> LLMNodeRunner: ...
    pass
class LLMEngine(LLMNode, LLMNodeBase):
    def __init__(self) -> None: ...
    def add_task_handler(self, inputs: typing.List[typing.Union[str, typing.Tuple[str, str]]], handler: LLMTaskHandler) -> None: ...
    def run(self, input_message: morpheus._lib.messages.ControlMessage) -> morpheus._lib.pycoro.CppToPyAwaitable: ...
    def run2(self, arg0: morpheus._lib.messages.ControlMessage) -> typing.Awaitable[typing.List[morpheus._lib.messages.ControlMessage]]: ...
    pass
class LLMNodeRunner():
    @property
    def inputs(self) -> typing.List[InputMap]:
        """
        :type: typing.List[InputMap]
        """
    @property
    def name(self) -> str:
        """
        :type: str
        """
    pass
class LLMTask():
    def __getitem__(self, arg0: str) -> object: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict) -> None: ...
    def __len__(self) -> int: ...
    def __setitem__(self, arg0: str, arg1: object) -> None: ...
    def get(self, arg0: str, arg1: object) -> object: ...
    @property
    def task_type(self) -> str:
        """
        :type: str
        """
    pass
class LLMTaskHandler():
    def __init__(self) -> None: ...
    def try_handle(self, context: LLMContext) -> morpheus._lib.pycoro.CppToPyAwaitable: ...
    pass
__version__ = '23.11.0'
