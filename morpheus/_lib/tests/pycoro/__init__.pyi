from __future__ import annotations
import morpheus._lib.tests.pycoro
import typing

__all__ = [
    "call_fib_async",
    "int_as_task"
]


def call_fib_async(arg0: object, arg1: int, arg2: int) -> typing.Awaitable[object]:
    pass
def int_as_task(arg0: int) -> typing.Awaitable[int]:
    pass
