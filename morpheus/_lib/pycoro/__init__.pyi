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
    "CppToPyAwaitable"
]


class CppToPyAwaitable():
    def __await__(self) -> CppToPyAwaitable: ...
    def __init__(self) -> None: ...
    def __iter__(self) -> CppToPyAwaitable: ...
    def __next__(self) -> None: ...
    pass
__version__ = '23.11.0'
