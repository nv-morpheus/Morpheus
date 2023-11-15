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


class _DecoratedSource(SingleOutputSource):

    def __init__(self, *gen_args, config: Config, gen_fn: typing.Callable, return_type: type, **gen_fn_kwargs):
        super().__init__(config)
        self._gen_fn = functools.partial(gen_fn, *gen_args, **gen_fn_kwargs)
        self._gen_fn_name = _get_name_from_fn(gen_fn)
        self._return_type = return_type

    @property
    def name(self) -> str:
        return self._gen_fn_name

    def supports_cpp_node(self) -> bool:
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(self._return_type)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        return builder.make_source(self.unique_name, self._gen_fn)


class _DecoratedSourceWithPreallocation(PreallocatorMixin, _DecoratedSource):
    pass


def _determine_return_type(gen_fn: typing.Callable) -> type:
    """
    Unpacks return types like:
    def soource() -> typing.Generator[MessageMeta, None, None]:
        ....
    """
    signature = inspect.signature(gen_fn)
    return_type = signature.return_annotation
    if return_type is signature.empty:
        raise ValueError(f"Return type of {gen_fn.__name__} has no type annotation, "
                         "required for functions using the @source decorator")

    # When someone uses collections.abc.Generator or collections.abc.Iterator the return type is an instance of
    # typing.GenericAlias, however when someone uses typing.Generator or typing.Iterator the return type is an
    # instance of typing._GenericAlias. We need to check for both.
    if isinstance(return_type, (typing.GenericAlias, typing._GenericAlias)):
        return_type = return_type.__args__[0]

    return return_type


def source(gen_fn: typing.Callable):

    @functools.wraps(gen_fn)
    def wrapper(config: Config, *args, **kwargs):
        return_type = _determine_return_type(gen_fn)

        # If the return type supports pre-allocation we use the pre-allocating source
        if return_type in (pd.DataFrame, cudf.DataFrame, MessageMeta, MultiMessage):
            return _DecoratedSourceWithPreallocation(*args,
                                                     config=config,
                                                     gen_fn=gen_fn,
                                                     return_type=return_type,
                                                     **kwargs)

        return _DecoratedSource(*args, config=config, gen_fn=gen_fn, return_type=return_type, **kwargs)

    return wrapper


class _DecoratedStage(SinglePortStage):

    def __init__(self, *on_data_args, config: Config, on_data_fn: typing.Callable, **on_data_kwargs):
        super().__init__(config)
        self._on_data_fn = functools.partial(on_data_fn, *on_data_args, **on_data_kwargs)
        self._on_data_fn_name = _get_name_from_fn(on_data_fn)

        signature = inspect.signature(on_data_fn)

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

    @functools.wraps(on_data_fn)
    def wrapper(config: Config, *args, **kwargs):
        return _DecoratedStage(*args, config=config, on_data_fn=on_data_fn, **kwargs)

    return wrapper
