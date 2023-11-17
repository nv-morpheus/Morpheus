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

import collections
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
GeneratorType = typing.Callable[..., collections.abc.Iterator[typing.Any]]


def _get_name_from_fn(fn: typing.Callable) -> str:
    try:
        return fn.__name__
    except AttributeError:
        # If the function is a partial, it won't have a name
        if isinstance(fn, functools.partial):
            return _get_name_from_fn(fn.func)

        return str(fn)


def _determine_return_type(gen_fn: GeneratorType) -> type:
    """
    Unpacks return type annoatations like:
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


def _is_dataframe_containing_type(type_: type) -> bool:
    return type_ in (pd.DataFrame, cudf.DataFrame, MessageMeta, MultiMessage)


class WrappedFunctionSourceStage(SingleOutputSource):
    """
    Source stage that wraps a generator function as the method for generating messages.

    The wrapped function must be a generator function. If `return_type` is not provided, the stage will use the
    return type annotation of `gen_fn` as the output type. If the return type annotation is not provided, the stage
    will use `typing.Any` as the output type.

    It is highly recommended to use specify either `return_type` or provide a return type annotation for `gen_fn`.

    If the output type is an instance of `pandas.DataFrame`, `cudf.DataFrame`, `MessageMeta`, or `MultiMessage` the
    `PreAllocatedWrappedFunctionStage` class should be used instead, as this will also perform any DataFrame column
    allocations needed by other stages in the pipeline.

    Any additional arguments passed in aside from `config` and `return_type`, will be bound to the wrapped function via
    `functools.partial`.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    gen_fn : `GeneratorType`
        Generator function to use as the source of messages.
    return_type : `type`, optional
        Return type of `gen_fn` if not provided the stage will use the return type annotation of `gen_fn` as the output
        if not provided the stage will use `typing.Any` as the output type.
    *gen_args : `typing.Any`
        Additional arguments to bind to `gen_fn` via `functools.partial`.
    **gen_fn_kwargs : `typing.Any`
        Additional keyword arguments to bind to `gen_fn` via `functools.partial`.
    """

    def __init__(self, config: Config, gen_fn: GeneratorType, *gen_args, return_type: type = None, **gen_fn_kwargs):
        super().__init__(config)
        # collections.abc.Generator is a subclass of collections.abc.Iterator
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
    """
    Source stage that wraps a generator function as the method for generating messages.

    This stage provides the same functionality as `WrappedFunctionSourceStage`, but also performs any DataFrame column
    allocations needed by other stages in the pipeline. As such the return type for `gen_fn` must be one of:
    `pandas.DataFrame`, `cudf.DataFrame`, `MessageMeta`, or `MultiMessage`.

    Any additional arguments passed in aside from `config` and `return_type`, will be bound to the wrapped function via
    `functools.partial`.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    gen_fn : `GeneratorType`
        Generator function to use as the source of messages.
    return_type : `type`, optional
        Return type of `gen_fn` if not provided the stage will use the return type annotation of `gen_fn` as the output.
    *gen_args : `typing.Any`
        Additional arguments to bind to `gen_fn` via `functools.partial`.
    **gen_fn_kwargs : `typing.Any`
        Additional keyword arguments to bind to `gen_fn` via `functools.partial`.
    """

    def __init__(self, config: Config, gen_fn: GeneratorType, *gen_args, return_type: type = None, **gen_fn_kwargs):
        super().__init__(*gen_args, config=config, gen_fn=gen_fn, return_type=return_type, **gen_fn_kwargs)
        if not _is_dataframe_containing_type(self._return_type):
            raise ValueError("PreAllocatedWrappedFunctionStage can only be used with DataFrame containing types")


def source(gen_fn: GeneratorType):
    """
    Decorator for wrapping a function as a source stage. The function must be a generator method.

    It is highly recommended to use a return type annotation, as this will be used by the stage as the output type. If
    no return type annotation is provided, the stage will use `typing.Any` as the output type.

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

    # Use wraps to ensure user's don't lose their function name and docstrinsgs, however we do want to override the
    # annotations to reflect that the returned function requires a config and returns a stage
    @functools.wraps(gen_fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def wrapper(config: Config, *args, **kwargs) -> WrappedFunctionSourceStage:
        return_type = _determine_return_type(gen_fn)

        # If the return type supports pre-allocation we use the pre-allocating source
        if _is_dataframe_containing_type(return_type):
            return PreAllocatedWrappedFunctionStage(*args,
                                                    config=config,
                                                    gen_fn=gen_fn,
                                                    return_type=return_type,
                                                    **kwargs)

        return WrappedFunctionSourceStage(*args, config=config, gen_fn=gen_fn, return_type=return_type, **kwargs)

    return wrapper


class WrappedFunctionStage(SinglePortStage):
    """
    Stage that wraps a function to be used for processing messages.

    The function must receive at least one argument, the first argument must be the incoming message, and must
    return a value. If `accept_type` is not provided, the type annotation of the first argument will be used, and if
    that parameter has no type annotation, the stage will be set to use `typing.Any` as the accept type.

    If `return_type` is not provided, the stage will use the return type annotation of `on_data_fn` as the output type.
    If the return type annotation is not provided, the stage will use the same type as the input.

    Any additional arguments passed in aside from `config`, `accept_type` and `return_type`, will be bound to the
    wrapped function via `functools.partial`.

    Parameters
    ----------
    config : `morpheus.config.Config`
        Pipeline configuration instance.
    on_data_fn : `typing.Callable`
        Function to be used for processing messages.
    accept_type : `type`, optional
        Type of message to accept, if not provided the stage will use the type annotation of the first parameter of
        `on_data_fn` as the accept type.
    return_type : `type`, optional
        Return type of `gen_fn` if not provided the stage will use the return type annotation of `gen_fn` as the output.
    *on_data_args : `typing.Any`
        Additional arguments to bind to `on_data_fn` via `functools.partial`.
    **on_data_kwargs : `typing.Any`
        Additional keyword arguments to bind to `on_data_fn` via `functools.partial`.
    """

    def __init__(self,
                 config: Config,
                 on_data_fn: typing.Callable,
                 *on_data_args,
                 accept_type: type = None,
                 return_type: type = None,
                 **on_data_kwargs):
        super().__init__(config)
        self._on_data_fn = functools.partial(on_data_fn, *on_data_args, **on_data_kwargs)
        self._on_data_fn_name = _get_name_from_fn(on_data_fn)

        # Even if both accept_type and return_type are provided, we should still need to inspect the function signature
        # to verify it is callable with at least one argument
        signature = inspect.signature(self._on_data_fn)

        try:
            first_param = next(iter(signature.parameters.values()))
            self._accept_type = accept_type or first_param.annotation
            if self._accept_type is signature.empty:
                logger.warning(
                    "%s argument of %s has no type annotation, defaulting to typing.Any for the stage accept type",
                    first_param.name,
                    self._on_data_fn_name)
                self._accept_type = typing.Any
        except StopIteration as e:
            raise ValueError(f"Wrapped stage functions {self._on_data_fn_name} must have at least one parameter") from e

        self._return_type = return_type or signature.return_annotation
        if self._return_type is signature.empty:
            logger.warning("Return type of %s has no type annotation, defaulting to the stage's accept type",
                           self._on_data_fn_name)
            self._return_type = self._accept_type

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
    Decorator for wrapping a function as a stage. The function must receive at least one argument, the first argument
    must be the incoming message, and must return a value.

    It is highly recommended to use type annotations for the function parameters and return type, as this will be used
    by the stage as the accept and output types. If the incoming message parameter has no type annotation, the stage
    will be use `typing.Any` as the input type. If the return type has no type annotation, the stage will be set to
    return the same type as the input type.

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

    # Use wraps to ensure user's don't lose their function name and docstrinsgs, however we do want to override the
    # annotations to reflect that the returned function requires a config and returns a stage
    @functools.wraps(on_data_fn, assigned=('__module__', '__name__', '__qualname__', '__doc__'))
    def wrapper(config: Config, *args, **kwargs) -> WrappedFunctionStage:
        return WrappedFunctionStage(*args, config=config, on_data_fn=on_data_fn, **kwargs)

    return wrapper
