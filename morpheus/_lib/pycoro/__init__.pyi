"""
        -----------------------
        .. currentmodule:: morpheus.llm
        .. autosummary::
           :toctree: _generate

        """
from __future__ import annotations
import morpheus._lib.pycoro
import typing

__all__ = [
    "CppToPyAwaitable",
    "wrap_coroutine"
]


class CppToPyAwaitable():
    def __await__(self) -> CppToPyAwaitable: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> CppToPyAwaitable: ...
    def __next__(self) -> None: ...
    pass
def wrap_coroutine(arg0: typing.Awaitable[typing.List[str]]) -> typing.Awaitable[str]:
    pass
