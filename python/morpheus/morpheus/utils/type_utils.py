# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

import inspect
import types
import typing
from collections import defaultdict

import numpy as np
import pandas as pd

from morpheus.config import ExecutionMode
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_aliases import DataFrameTypeStr

# pylint: disable=invalid-name
T_co = typing.TypeVar("T_co", covariant=True)

# pylint: disable=invalid-name
T = typing.TypeVar('T')
T1 = typing.TypeVar('T1')
T2 = typing.TypeVar('T2')
T3 = typing.TypeVar('T3')
T4 = typing.TypeVar('T4')

# Use _DecoratorType as a type variable for decorators. See:
# https://github.com/python/mypy/pull/8336/files#diff-eb668b35b7c0c4f88822160f3ca4c111f444c88a38a3b9df9bb8427131538f9cR260
_DecoratorType = typing.TypeVar("_DecoratorType", bound=typing.Callable[..., typing.Any])

# pylint: enable=invalid-name


def greatest_ancestor(*cls_list):
    """
    Returns the greatest common ancestor of the classes in the class list
    """
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


def is_union_type(type_: type) -> bool:
    """
    Returns True if the type is a `typing.Union` or a `types.UnionType`.
    """
    # Unions in the form of `(float | int)` are instances of `types.UnionType`.
    # However, unions in the form of `typing.Union[float, int]` are instances of `typing._UnionGenericAlias`.
    return isinstance(type_, (types.UnionType, typing._UnionGenericAlias))


def flatten_types(type_list: list[type]) -> list[type]:
    """
    Flattens a list of types, removing any union and `typing.Any` types.
    """
    flattened_types = []
    for type_ in type_list:
        if type_ is typing.Any:
            type_ = object

        if is_union_type(type_):
            flattened_types.extend(typing.get_args(type_))
        else:
            flattened_types.append(type_)


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

    out_union = None

    if (len(cls_list) == 1):
        return typing.Union[cls_list[0]]

    out_union = unpack_union(cls_list[0:2])

    # Since typing.Union[typing.Union[A, B], C] == typing.Union[A, B, C], we build the union up manually
    for type_ in cls_list[2:]:
        out_union = typing.Union[out_union, type_]

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

    out_tuple = None

    if (len(cls_list) == 1):
        return typing.Tuple[cls_list[0]]

    out_tuple = unpack_tuple(cls_list[0:2])

    # Since typing.Tuple[typing.Tuple[A, B], C] == typing.Tuple[A, B, C], we build the union up manually
    for type_ in cls_list[2:]:
        out_tuple = typing.Tuple[out_tuple, type_]

    return out_tuple


def pretty_print_type_name(type_: type) -> str:
    """
    Determines a good label to use for a type. Keeps the strings shorter.
    """

    if (type_.__module__ == "typing"):
        return str(type_).replace("typing.", "")

    return type_.__module__.split(".")[0] + "." + type_.__name__


def get_full_qualname(klass: type) -> str:
    """
    Returns the fully qualified name of a class.
    """
    module = klass.__module__
    if module == '__builtin__':
        return klass.__qualname__
    return module + '.' + klass.__qualname__


def df_type_str_to_exec_mode(df_type_str: DataFrameTypeStr) -> ExecutionMode:
    """
    Return the appropriate execution mode based on the DataFrame type string.
    """
    if df_type_str == "cudf":
        return ExecutionMode.GPU
    if df_type_str == "pandas":
        return ExecutionMode.CPU

    valid_values = ", ".join(typing.get_args(DataFrameTypeStr))
    raise ValueError(f"Invalid DataFrame type string: {df_type_str}, valid values are: {valid_values}")


def exec_mode_to_df_type_str(execution_mode: ExecutionMode) -> DataFrameTypeStr:
    if execution_mode == ExecutionMode.GPU:
        return "cudf"

    return "pandas"


def df_type_str_to_pkg(df_type_str: DataFrameTypeStr) -> types.ModuleType:
    """
    Return the appropriate DataFrame package based on the DataFrame type string.
    """
    if df_type_str == "cudf":
        import cudf
        return cudf
    if df_type_str == "pandas":
        return pd

    valid_values = ", ".join(typing.get_args(DataFrameTypeStr))
    raise ValueError(f"Invalid DataFrame type string: {df_type_str}, valid values are: {valid_values}")


@typing.overload
def get_df_pkg(selector: DataFrameTypeStr) -> types.ModuleType:
    ...


@typing.overload
def get_df_pkg(selector: ExecutionMode) -> types.ModuleType:
    ...


def get_df_pkg(selector: ExecutionMode | DataFrameTypeStr) -> types.ModuleType:
    """
    Return the appropriate DataFrame package based on the execution mode.
    """
    if not isinstance(selector, ExecutionMode):
        execution_mode = df_type_str_to_exec_mode(selector)
    else:
        execution_mode = selector

    if execution_mode == ExecutionMode.GPU:
        import cudf
        return cudf

    return pd


def get_array_pkg(execution_mode: ExecutionMode) -> types.ModuleType:
    """
    Return the appropriate array package (CuPy for GPU, NumPy for CPU) based on the execution mode.
    """

    if execution_mode == ExecutionMode.GPU:
        import cupy
        return cupy

    return np


@typing.overload
def get_df_class(df_type_str: DataFrameTypeStr) -> type[DataFrameType]:
    ...


def get_df_class(execution_mode: ExecutionMode) -> type[DataFrameType]:
    """
    Return the appropriate DataFrame class based on the execution mode.
    """
    df_pkg = get_df_pkg(execution_mode)
    return df_pkg.DataFrame


def is_cudf_type(obj: typing.Any) -> bool:
    """
    Check if a given object (DataFrame, Series, RangeIndex etc...) is a cuDF type.
    """
    return "cudf" in str(type(obj))


def get_df_pkg_from_obj(obj: typing.Any) -> types.ModuleType:
    """
    Return the appropriate DataFrame package based on the DataFrame object.
    """
    if is_cudf_type(obj):
        import cudf
        return cudf

    return pd


def is_dataframe(obj: typing.Any) -> bool:
    """
    Check if a given object is a pandas or cudf DataFrame.
    """
    df_pkg = get_df_pkg_from_obj(obj)
    return isinstance(obj, df_pkg.DataFrame)
