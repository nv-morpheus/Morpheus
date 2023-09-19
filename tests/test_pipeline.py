#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import typing

import pytest

from _utils import assert_results
from _utils.stages.conv_msg import ConvMsg
from _utils.stages.multi_message_pass_thru import MultiMessagePassThruStage
from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.type_aliases import DataFrameType


class SourceTestStage(InMemorySourceStage):

    def __init__(self,
                 config,
                 dataframes: typing.List[DataFrameType],
                 destructor_cb: typing.Callable[[], None] = None,
                 repeat: int = 1):
        super().__init__(config, dataframes, repeat)
        self._destructor_cb = destructor_cb

    @property
    def name(self) -> str:
        return "test-source"

    def __del__(self):
        if self._destructor_cb is not None:
            self._destructor_cb()


class SinkTestStage(InMemorySinkStage):

    def __init__(self,
                 config,
                 on_start_cb: typing.Callable[[], None] = None,
                 start_async_cb: typing.Callable[[], None] = None,
                 destructor_cb: typing.Callable[[], None] = None):
        super().__init__(config)
        self._on_start_cb = on_start_cb
        self._start_async_cb = start_async_cb
        self._destructor_cb = destructor_cb

    @property
    def name(self) -> str:
        return "test-sink"

    def on_start(self):
        if self._on_start_cb is not None:
            self._on_start_cb()

    async def start_async(self):
        await super().start_async()
        if self._start_async_cb is not None:
            self._start_async_cb()

    def __del__(self):
        if self._destructor_cb is not None:
            self._destructor_cb()


def _run_pipeline(filter_probs_df: DataFrameType,
                  source_callbacks: dict[str, typing.Callable[[], None]],
                  sink_callbacks: dict[str, typing.Callable[[], None]]):
    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(SourceTestStage(config, [filter_probs_df], **source_callbacks))
    pipe.add_stage(SinkTestStage(config, **sink_callbacks))
    pipe.run()


@pytest.mark.use_cudf
def test_destructors_called(filter_probs_df: DataFrameType):
    """
    Test to ensure that the destructors of stages are called (issue #1114).
    """
    state_dict = {"source": False, "sink": False}

    def update_state_dict(key: str):
        nonlocal state_dict
        state_dict[key] = True

    source_callbacks = {'destructor_cb': lambda: update_state_dict("source")}
    sink_callbacks = {'destructor_cb': lambda: update_state_dict("sink")}

    _run_pipeline(filter_probs_df, source_callbacks, sink_callbacks)

    gc.collect()
    assert state_dict["source"]
    assert state_dict["sink"]


@pytest.mark.use_cudf
def test_startup_cb_called(filter_probs_df: DataFrameType):
    """
    Test to ensure that the destructors of stages are called (issue #1114).
    """
    state_dict = {"on_start": False, "start_async": False}

    def update_state_dict(key: str):
        nonlocal state_dict
        state_dict[key] = True

    sink_callbacks = {
        'on_start_cb': lambda: update_state_dict("on_start"),
        'start_async_cb': lambda: update_state_dict("start_async")
    }

    _run_pipeline(filter_probs_df, source_callbacks={}, sink_callbacks=sink_callbacks)

    assert state_dict["on_start"]
    assert state_dict["start_async"]


@pytest.mark.use_cudf
def test_pipeline_narrowing_types(config: Config, filter_probs_df: DataFrameType):
    """
    Test to ensure that we aren't narrowing the types of messages in the pipeline.

    In this case, `ConvMsg` emits `MultiResponseMessage` messages which are a subclass of `MultiMessage`,
    which is the accepted type for `MultiMessagePassThruStage`. We want to ensure that the type is retained allowing us
    to place a stage after `MultiMessagePassThruStage` requring `MultiResponseMessage` like `AddScoresStage`.
    """
    config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
    expected_df = filter_probs_df.to_pandas()
    expected_df = expected_df.rename(columns=dict(zip(expected_df.columns, config.class_labels)))

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(ConvMsg(config))
    pipe.add_stage(MultiMessagePassThruStage(config))
    pipe.add_stage(AddScoresStage(config))
    pipe.add_stage(SerializeStage(config, include=[f"^{c}$" for c in config.class_labels]))
    compare_stage = pipe.add_stage(CompareDataFrameStage(config, compare_df=expected_df))
    pipe.run()

    assert_results(compare_stage.get_results())
