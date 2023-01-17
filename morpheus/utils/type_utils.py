# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import functools
import inspect
import typing
from collections import defaultdict

T_co = typing.TypeVar("T_co", covariant=True)
T = typing.TypeVar('T')
T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')
T3 = typing.TypeVar('T3')
T4 = typing.TypeVar('T4')

# Use _DecoratorType as a type variable for decorators. See:
# https://github.com/python/mypy/pull/8336/files#diff-eb668b35b7c0c4f88822160f3ca4c111f444c88a38a3b9df9bb8427131538f9cR260
_DecoratorType = typing.TypeVar("_DecoratorType", bound=typing.Callable[..., typing.Any])


def greatest_ancestor(*cls_list):
    mros = [list(inspect.getmro(cls)) for cls in cls_list]
    track = defaultdict(int)
    while mros:
        for mro in mros:
            cur = mro.pop(0)
            track[cur] += 1
            if track[cur] == len(cls_list):
                return cur
            if len(mro) == 0:
                mros.remove(mro)
    return None  # or raise, if that's more appropriate


@typing.overload
def unpack_union(cls_1: typing.Type[T]) -> typing.Union[typing.Type[T]]:
    ...


@typing.overload
def unpack_union(cls_1: typing.Type[T1], cls_2: typing.Type[T2]) -> typing.Union[typing.Type[T1], typing.Type[T2]]:
    ...


@typing.overload
def unpack_union(cls_1: typing.Type[T1], cls_2: typing.Type[T2],
                 cls_3: typing.Type[T3]) -> typing.Union[typing.Type[T1], typing.Type[T2], typing.Type[T3]]:
    ...


def unpack_union(*cls_list: typing.Type) -> typing.Union:

    assert len(cls_list) > 0, "Union class list must have at least 1 element."

    if (len(cls_list) == 1):
        return typing.Union[cls_list[0]]
    # elif (len(cls_list) == 2):
    #     return typing.Union[cls_list[0], cls_list[1]]
    else:
        out_union = unpack_union(cls_list[0:2])

        # Since typing.Union[typing.Union[A, B], C] == typing.Union[A, B, C], we build the union up manually
        for t in cls_list[2:]:
            out_union = typing.Union[out_union, t]

        return out_union


@typing.overload
def unpack_tuple(cls_1: typing.Type[T]) -> typing.Tuple[typing.Type[T]]:
    ...


@typing.overload
def unpack_tuple(cls_1: typing.Type[T1], cls_2: typing.Type[T2]) -> typing.Tuple[typing.Type[T1], typing.Type[T2]]:
    ...


@typing.overload
def unpack_tuple(cls_1: typing.Type[T1], cls_2: typing.Type[T2],
                 cls_3: typing.Type[T3]) -> typing.Tuple[typing.Type[T1], typing.Type[T2], typing.Type[T3]]:
    ...


def unpack_tuple(*cls_list: typing.Type) -> typing.Tuple:

    assert len(cls_list) > 0, "Union class list must have at least 1 element."

    if (len(cls_list) == 1):
        return typing.Tuple[cls_list[0]]
    # elif (len(cls_list) == 2):
    #     return typing.Union[cls_list[0], cls_list[1]]
    else:
        out_tuple = unpack_tuple(cls_list[0:2])

        # Since typing.Tuple[typing.Tuple[A, B], C] == typing.Tuple[A, B, C], we build the union up manually
        for t in cls_list[2:]:
            out_tuple = typing.Tuple[out_tuple, t]

        return out_tuple


def pretty_print_type_name(t: typing.Type) -> str:
    """
    Determines a good label to use for a type. Keeps the strings shorter.
    """

    if (t.__module__ == "typing"):
        return str(t).replace("typing.", "")

    return t.__module__.split(".")[0] + "." + t.__name__


def mirror_args(wrapped: _DecoratorType,
                assigned=('__doc__', '__annotations__'),
                updated=functools.WRAPPER_UPDATES) -> typing.Callable[[_DecoratorType], _DecoratorType]:
    return functools.wraps(wrapped=wrapped, assigned=assigned, updated=updated)


def get_full_qualname(klass: typing.Type) -> str:
    module = klass.__module__
    if module == '__builtin__':
        return klass.__qualname__
    return module + '.' + klass.__qualname__
