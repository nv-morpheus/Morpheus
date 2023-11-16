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
import typing

import pandas as pd
import pytest

import cudf

from _utils import assert_results
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.pipeline.stage_decorator import PreAllocatedWrappedFunctionStage
from morpheus.pipeline.stage_decorator import WrappedFunctionSourceStage
from morpheus.pipeline.stage_decorator import WrappedFunctionStage
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage


@pytest.mark.use_python
@pytest.mark.parametrize("generator_type",
                         [None, typing.Iterator, typing.Generator, collections.abc.Iterator, collections.abc.Generator])
@pytest.mark.parametrize("return_type, is_prealloc",
                         [(pd.DataFrame, True), (cudf.DataFrame, True), (MessageMeta, True), (MultiMessage, True),
                          (float, False)])
def test_source_decorator(config: Config, generator_type: type, return_type: type, is_prealloc: bool):
    if generator_type is not None:
        if generator_type in (typing.Generator, collections.abc.Generator):
            return_annotation = generator_type[return_type, None, None]
        else:
            return_annotation = generator_type[return_type]
    else:
        return_annotation = return_type

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
