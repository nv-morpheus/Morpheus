"""
        -----------------------
        .. currentmodule:: morpheus.common
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        """
from __future__ import annotations
import morpheus._lib.common
import typing

__all__ = [
    "FiberQueue",
    "Tensor"
]


class FiberQueue():
    def __init__(self, max_size: int) -> None: ...
    def close(self) -> None: ...
    def get(self, block: bool = True, timeout: float = 0.0) -> object: ...
    def put(self, item: object, block: bool = True, timeout: float = 0.0) -> None: ...
    pass
class Tensor():
    @property
    def __cuda_array_interface__(self) -> dict:
        """
        :type: dict
        """
    pass
__version__ = 'dev'
