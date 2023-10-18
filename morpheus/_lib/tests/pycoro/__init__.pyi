from __future__ import annotations
import morpheus._lib.tests.pycoro
import typing

__all__ = [
    "call_async",
    "call_fib_async",
    "raise_at_depth_async"
]


def call_async(arg0: object) -> typing.Awaitable[object]:
    pass
def call_fib_async(arg0: object, arg1: int, arg2: int) -> typing.Awaitable[object]:
    pass
def raise_at_depth_async(arg0: object, arg1: int) -> typing.Awaitable[object]:
    pass
