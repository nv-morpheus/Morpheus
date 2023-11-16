# Copyright (c) 2023, NVIDIA CORPORATION.
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
import logging
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema

logger = logging.getLogger(__name__)


def _get_name_from_fn(fn: typing.Callable) -> str:
    try:
        return fn.__name__
    except AttributeError:
        # If the function is a partial, it won't have a name
        return str(fn)


def _determine_return_type(gen_fn: typing.Callable) -> type:
    """
    Unpacks return types like:
    def soource() -> typing.Generator[MessageMeta, None, None]:
        ....
    """
    signature = inspect.signature(gen_fn)
    return_type = signature.return_annotation
    if return_type is signature.empty:
        return_type = typing.Any

    # When someone uses collections.abc.Generator or collections.abc.Iterator the return type is an instance of
    # typing.GenericAlias, however when someone uses typing.Generator or typing.Iterator the return type is an
    # instance of typing._GenericAlias. We need to check for both.
    if isinstance(return_type, (typing.GenericAlias, typing._GenericAlias)):
        return_type = return_type.__args__[0]

    return return_type


class WrappedFunctionSourceStage(SingleOutputSource):

    def __init__(self, *gen_args, config: Config, gen_fn: typing.Callable, return_type: type = None, **gen_fn_kwargs):
        super().__init__(config)
        if not inspect.isgeneratorfunction(gen_fn):
            raise ValueError("Wrapped source functions must be generator functions")

        self._gen_fn = functools.partial(gen_fn, *gen_args, **gen_fn_kwargs)
        self._gen_fn_name = _get_name_from_fn(gen_fn)
        self._return_type = return_type or _determine_return_type(gen_fn)

    @property
    def name(self) -> str:
        return self._gen_fn_name

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(self._return_type)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._gen_fn)


class PreAllocatedWrappedFunctionStage(PreallocatorMixin, WrappedFunctionSourceStage):
    pass


def source(gen_fn: typing.Callable):
    """
    Decorator for wrapping a function as a source stage. The function must be a generator method.

    It is highly recommended to use a type annotation for the return type, as this will be used by the stage as the
    output type. If no return type annotation is provided, the stage will use `typing.Any` as the output type.

    When invoked the wrapped function will return a source stage, any additional arguments passed in aside from the
    config, will be bound to the wrapped function via `functools.partial`.

    Examples
    --------

    >>> @source
    ... def source_gen(dataframes: list[cudf.DataFrame]) -> collections.abc.Iterator[MessageMeta]:
    ...     for df in dataframes:
    ...         yield MessageMeta(df)
    ...
    >>>

    >>> pipe.set_source(source_gen(config, dataframes=[df]))
    """

    @functools.wraps(gen_fn)
    def wrapper(config: Config, *args, **kwargs):
        return_type = _determine_return_type(gen_fn)

        # If the return type supports pre-allocation we use the pre-allocating source
        if return_type in (pd.DataFrame, cudf.DataFrame, MessageMeta, MultiMessage):
            return PreAllocatedWrappedFunctionStage(*args,
                                                    config=config,
                                                    gen_fn=gen_fn,
                                                    return_type=return_type,
                                                    **kwargs)

        return WrappedFunctionSourceStage(*args, config=config, gen_fn=gen_fn, return_type=return_type, **kwargs)

    return wrapper


class WrappedFunctionStage(SinglePortStage):

    def __init__(self, *on_data_args, config: Config, on_data_fn: typing.Callable, **on_data_kwargs):
        super().__init__(config)
        self._on_data_fn = functools.partial(on_data_fn, *on_data_args, **on_data_kwargs)
        self._on_data_fn_name = _get_name_from_fn(on_data_fn)

        signature = inspect.signature(self._on_data_fn)

        try:
            first_param = next(iter(signature.parameters.values()))
            self._accept_type = first_param.annotation
            if self._accept_type is signature.empty:
                logger.warning(
                    "%s argument of %s has no type annotation, defaulting to typing.Any for the stage accept type",
                    first_param.name,
                    self._on_data_fn_name)
                self._accept_type = typing.Any

            self._return_type = signature.return_annotation
            if self._return_type is signature.empty:
                logger.warning("Return type of %s has no type annotation, defaulting to the stage's accept type",
                               self._on_data_fn_name)
                self._return_type = self._accept_type

        except StopIteration as e:
            raise ValueError(f"Wrapped stage functions {self._on_data_fn_name} must have at least one parameter") from e

    @property
    def name(self) -> str:
        return self._on_data_fn_name

    def accepted_types(self) -> typing.Tuple:
        return (self._accept_type, )

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        if self._return_type is not typing.Any:
            return_type = self._return_type
        else:
            return_type = schema.input_schema.get_type()

        schema.output_schema.set_type(return_type)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self._on_data_fn))
        builder.make_edge(input_node, node)

        return node


def stage(on_data_fn: typing.Callable):
    """
    Decorator for wrapping a function as a stage. The function must receive at least one argument, and the first
    argument must be the incoming message, and must return a value.

    It is highly recommended to use type annotations for the function parameters and return type, as this will be used
    by the stage as the send and receive types. If the incoming message parameter has no type annotation, the stage
    will be set to accept `typing.Any` as the input type. If the return type has no type annotation, the stage will
    be set to return the same type as the input type.

    When invoked the wrapped function will return a stage, any additional arguments passed in aside from the config,
    will be bound to the wrapped function via `functools.partial`.

    Examples
    --------

    >>> @stage
    ... def multiplier(message: MessageMeta, column: str, value: int | float) -> MessageMeta:
    ...     with message.mutable_dataframe() as df:
    ...         df[column] = df[column] * value
    ...
    ...     return message
    ...
    >>>

    >>> pipe.add_stage(multiplier(config, column='v2', value=5))
    """

    @functools.wraps(on_data_fn)
    def wrapper(config: Config, *args, **kwargs):
        return WrappedFunctionStage(*args, config=config, on_data_fn=on_data_fn, **kwargs)

    return wrapper
