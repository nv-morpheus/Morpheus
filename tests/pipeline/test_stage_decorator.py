#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import functools
import inspect
import typing

import pandas as pd
import pytest

import cudf

from _utils import assert_results
from morpheus.common import TypeId
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import PreAllocatedWrappedFunctionStage
from morpheus.pipeline.stage_decorator import WrappedFunctionSourceStage
from morpheus.pipeline.stage_decorator import WrappedFunctionStage
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


def _get_annotation(type_: type, generator_type: type) -> type:
    if generator_type is not None:
        if generator_type in (typing.Generator, collections.abc.Generator):
            annotation = generator_type[type_, None, None]
        else:
            annotation = generator_type[type_]
    else:
        annotation = type_

    return annotation


@pytest.mark.use_python
@pytest.mark.parametrize("generator_type",
                         [None, typing.Iterator, typing.Generator, collections.abc.Iterator, collections.abc.Generator])
@pytest.mark.parametrize("return_type, is_prealloc",
                         [(pd.DataFrame, True), (cudf.DataFrame, True), (MessageMeta, True), (MultiMessage, True),
                          (float, False)])
def test_wrapped_function_source_stage_constructor(config: Config,
                                                   generator_type: type,
                                                   return_type: type,
                                                   is_prealloc: bool):
    return_annotation = _get_annotation(return_type, generator_type)

    def test_source_gen() -> return_annotation:
        yield None

    if is_prealloc:
        source_cls = PreAllocatedWrappedFunctionStage
    else:
        source_cls = WrappedFunctionSourceStage

    source_stage = source_cls(config, gen_fn=test_source_gen)

    assert isinstance(source_stage, WrappedFunctionSourceStage)
    assert is_prealloc == isinstance(source_stage, PreAllocatedWrappedFunctionStage)

    # check the output type
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)  # pylint: disable=no-member
    assert schema.output_schema.get_type() is return_type


@pytest.mark.use_python
@pytest.mark.parametrize("src_cls", [WrappedFunctionSourceStage, PreAllocatedWrappedFunctionStage])
@pytest.mark.parametrize("use_partial", [True, False])
def test_wrapped_function_source_stage_name(config: Config, src_cls: type, use_partial: bool):

    def test_source_gen(value: int) -> cudf.DataFrame:
        yield value

    if use_partial:
        source_stage = src_cls(config, functools.partial(test_source_gen, value=5))
    else:
        source_stage = src_cls(config, test_source_gen, value=5)
    assert source_stage.name == 'test_source_gen'


@pytest.mark.use_python
@pytest.mark.parametrize("src_cls", [WrappedFunctionSourceStage, PreAllocatedWrappedFunctionStage])
def test_wrapped_function_stage_not_generator_error(config: Config, src_cls: type):

    def test_source_gen() -> MessageMeta:
        return MessageMeta(cudf.DataFrame())

    with pytest.raises(ValueError):
        src_cls(config, test_source_gen)


@pytest.mark.use_python
@pytest.mark.parametrize("return_type", [float, int, str, bool])
def test_pre_allocated_wrapped_function_stage_not_df_error(config: Config, return_type: type):

    def test_source_gen() -> return_type:
        yield None

    with pytest.raises(ValueError):
        PreAllocatedWrappedFunctionStage(config, test_source_gen)


@pytest.mark.use_python
@pytest.mark.parametrize("generator_type",
                         [None, typing.Iterator, typing.Generator, collections.abc.Iterator, collections.abc.Generator])
@pytest.mark.parametrize("return_type, is_prealloc",
                         [(pd.DataFrame, True), (cudf.DataFrame, True), (MessageMeta, True), (MultiMessage, True),
                          (float, False)])
def test_source_decorator(config: Config, generator_type: type, return_type: type, is_prealloc: bool):
    return_annotation = _get_annotation(return_type, generator_type)

    @source
    def test_source_gen() -> return_annotation:
        yield None

    source_stage = test_source_gen(config)  # pylint: disable=too-many-function-args

    assert isinstance(source_stage, WrappedFunctionSourceStage)
    assert is_prealloc == isinstance(source_stage, PreAllocatedWrappedFunctionStage)

    # check the output type
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)  # pylint: disable=no-member
    assert schema.output_schema.get_type() is return_type


@pytest.mark.use_python
def test_source_decorator_name(config: Config):

    @source
    def test_source_gen(value: int) -> int:
        yield value

    source_stage = test_source_gen(config, value=5)  # pylint: disable=redundant-keyword-arg
    assert source_stage.name == 'test_source_gen'  # pylint: disable=no-member


@pytest.mark.use_python
def test_source_decorator_no_annoation(config: Config):

    @source
    def test_source_gen():
        yield None

    source_stage = test_source_gen(config)  # pylint: disable=too-many-function-args

    assert isinstance(source_stage, WrappedFunctionSourceStage)
    assert not isinstance(source_stage, PreAllocatedWrappedFunctionStage)

    # check the output type
    schema = StageSchema(source_stage)
    source_stage.compute_schema(schema)  # pylint: disable=no-member
    assert schema.output_schema.get_type() is typing.Any


@pytest.mark.use_python
def test_not_generator_error(config: Config):

    @source
    def test_fn() -> int:
        return 5

    with pytest.raises(ValueError):
        test_fn(config)  # pylint: disable=too-many-function-args


@pytest.mark.use_python
@pytest.mark.parametrize("use_annotations", [True, False])
@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, None), (float, typing.Any), (None, None),
                          (None, float), (typing.Any, float), (typing.Any, typing.Any)])
def test_wrapped_function_stage_constructor(config: Config, use_annotations: bool, accept_type: type,
                                            return_type: type):

    if accept_type is None:
        accept_annotation = inspect.Signature.empty
        expected_accept_type = typing.Any
    else:
        accept_annotation = accept_type
        expected_accept_type = accept_type

    if return_type is None:
        return_annotation = inspect.Signature.empty
        expected_return_type = expected_accept_type
    else:
        return_annotation = return_type
        expected_return_type = return_type

    if use_annotations:

        def test_fn(message: accept_annotation) -> return_annotation:
            return message

        kwargs = {'accept_type': None, 'return_type': None}
    else:

        def test_fn(message):
            return message

        kwargs = {'accept_type': accept_type, 'return_type': return_type}

    wrapped_stage = WrappedFunctionStage(config, on_data_fn=test_fn, **kwargs)

    assert isinstance(wrapped_stage, WrappedFunctionStage)
    assert wrapped_stage.accepted_types() == (expected_accept_type, )
    assert wrapped_stage._return_type is expected_return_type


@pytest.mark.use_python
@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, None), (float, typing.Any), (None, None),
                          (None, float), (typing.Any, float), (typing.Any, typing.Any)])
def test_wrapped_function_stage_output_types(config: Config, accept_type: type, return_type: type):
    # For non-source types we need an upstream before we can check the compute_schema method outside of a pipeline
    if accept_type is None:
        expected_accept_type = typing.Any
    else:
        expected_accept_type = accept_type

    if return_type is None:
        expected_return_type = expected_accept_type
    elif return_type is typing.Any:
        expected_return_type = expected_accept_type
    else:
        expected_return_type = return_type

    wrapped_stage = WrappedFunctionStage(config,
                                         on_data_fn=lambda x: x,
                                         accept_type=accept_type,
                                         return_type=return_type)

    def source_fn():
        yield None

    upstream = WrappedFunctionSourceStage(config, source_fn, return_type=expected_accept_type)

    pipe = LinearPipeline(config)
    pipe.set_source(upstream)
    pipe.add_stage(wrapped_stage)
    pipe.build()
    schema = StageSchema(wrapped_stage)
    wrapped_stage.compute_schema(schema)
    assert schema.output_schema.get_type() is expected_return_type


@pytest.mark.use_python
@pytest.mark.parametrize("use_partial", [True, False])
def test_wrapped_function_stage_name(config: Config, use_partial: bool):

    def multiplier(message: MessageMeta, column: str, value: int | float) -> MessageMeta:
        with message.mutable_dataframe() as df:
            df[column] = df[column] * value

        return message

    if use_partial:
        wrapped_stage = WrappedFunctionStage(config, functools.partial(multiplier, column='v2', value=5))
    else:
        wrapped_stage = WrappedFunctionStage(config, multiplier, column='v2', value=5)
    assert wrapped_stage.name == 'multiplier'


@pytest.mark.use_python
def test_wrapped_function_stage_no_name(config: Config):
    # Class instances don't have a __name__ attribute like functions do
    class CallableCls:

        def __call__(self, message: MessageMeta) -> MessageMeta:
            return message

    wrapped_stage = WrappedFunctionStage(config, CallableCls())

    # The impl will fall back to calling str() on the object, but for our purposes we just want to make sure we got a
    # non-empty string
    assert isinstance(wrapped_stage.name, str)
    assert len(wrapped_stage.name) > 0


@pytest.mark.use_python
@pytest.mark.parametrize("needed_columns",
                         [None, {
                             'result': TypeId.INT64
                         }, {
                             'a': TypeId.INT64, 'b': TypeId.FLOAT32, 'c': TypeId.STRING
                         }])
def test_wrapped_function_stage_needed_columns(config: Config, needed_columns: dict[str, TypeId]):

    def test_fn(message: MessageMeta) -> MessageMeta:
        return message

    wrapped_stage = WrappedFunctionStage(config, test_fn, needed_columns=needed_columns)
    expected_needed_columns = needed_columns or collections.OrderedDict()
    assert wrapped_stage._needed_columns == expected_needed_columns


@pytest.mark.use_python
@pytest.mark.parametrize("accept_type, return_type",
                         [(pd.DataFrame, MessageMeta), (int, int), (MessageMeta, MessageMeta), (typing.Any, bool),
                          (typing.Union[float, int], float), (float, None), (float, typing.Any), (None, None),
                          (None, float), (typing.Any, float), (typing.Any, typing.Any)])
def test_stage_decorator(config: Config, accept_type: type, return_type: type):

    if accept_type is None:
        accept_annotation = inspect.Signature.empty
        expected_accept_type = typing.Any
    else:
        accept_annotation = accept_type
        expected_accept_type = accept_type

    if return_type is None:
        return_annotation = inspect.Signature.empty
        expected_return_type = expected_accept_type
    else:
        return_annotation = return_type
        expected_return_type = return_type

    @stage
    def test_fn(message: accept_annotation) -> return_annotation:
        return message

    wrapped_stage = test_fn(config)

    assert isinstance(wrapped_stage, WrappedFunctionStage)
    assert wrapped_stage.accepted_types() == (expected_accept_type, )
    assert wrapped_stage._return_type is expected_return_type


@pytest.mark.use_python
def test_stage_decorator_name(config: Config):

    @stage
    def test_fn(message: float, value: float) -> float:
        return message * value

    wrapped_stage = test_fn(config, value=2.2)
    assert wrapped_stage.name == 'test_fn'


@pytest.mark.use_python
def test_stage_decorator_no_annotation(config: Config):

    @stage
    def test_fn(message):
        return message

    wrapped_stage = test_fn(config)

    assert wrapped_stage.accepted_types() == (typing.Any, )
    assert wrapped_stage._return_type is typing.Any


@pytest.mark.use_python
@pytest.mark.parametrize("needed_columns",
                         [None, {
                             'result': TypeId.INT64
                         }, {
                             'a': TypeId.INT64, 'b': TypeId.FLOAT32, 'c': TypeId.STRING
                         }])
def test_stage_decorator_needed_columns(config: Config, needed_columns: dict[str, TypeId]):

    @stage(needed_columns=needed_columns)
    def test_fn(message: MessageMeta) -> MessageMeta:
        return message

    wrapped_stage = test_fn(config)
    expected_needed_columns = needed_columns or collections.OrderedDict()
    assert wrapped_stage._needed_columns == expected_needed_columns


def test_end_to_end_pipe(config: Config, filter_probs_df: cudf.DataFrame):

    @source
    def source_gen(dataframes: list[cudf.DataFrame]) -> collections.abc.Iterator[MessageMeta]:
        for df in dataframes:
            yield MessageMeta(df)

    @stage
    def multiplier(message: MessageMeta, column: str, value: int | float) -> MessageMeta:
        with message.mutable_dataframe() as df:
            df[column] = df[column] * value

        return message

    multipy_by = 5
    expected_df = filter_probs_df.copy(deep=True)
    expected_df['v2'] = expected_df['v2'] * multipy_by

    pipe = LinearPipeline(config)
    pipe.set_source(source_gen(config, dataframes=[filter_probs_df]))  # pylint: disable=redundant-keyword-arg
    pipe.add_stage(multiplier(config, column='v2', value=multipy_by))
    sink = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(sink.get_results())
