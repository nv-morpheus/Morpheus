#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections.abc
import os
import time
import typing

import mrc
import pytest

from _utils.stages.in_memory_source_x_stage import InMemSourceXStage
from _utils.stages.record_thread_id_stage import RecordThreadIdStage
from morpheus.config import Config
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage


@source
def error_source(subscription: mrc.Subscription, *, raise_error: bool = False) -> collections.abc.Iterator[int]:
    yield 1

    if raise_error:
        raise RuntimeError("Test error in source")

    while subscription.is_subscribed():
        time.sleep(0.1)


@stage
def error_stage(i: int, *, raise_error: bool = False) -> int:
    if raise_error:
        raise RuntimeError("Test error in stage")

    return i


@pytest.mark.parametrize("source_error, stage_error", [(True, False), (False, True), (True, True)])
def test_pipeline(config: Config, source_error: bool, stage_error: bool):
    """
    When source_error=False and stage_error=True this reproduces issue #1838
    """

    config = Config()
    pipe = LinearPipeline(config)
    pipe.set_source(error_source(config, raise_error=source_error))
    pipe.add_stage(error_stage(config, raise_error=stage_error))

    with pytest.raises(RuntimeError, match="^Test error in (source|stage)$"):
        pipe.run()


class SimpleSource(InMemSourceXStage):
    """
    InMemorySourceStage subclass that emits whatever you give it and doesn't assume the source data
    is a dataframe.
    """

    def __init__(self, c: Config, data: list[typing.Any], pe_count: int):
        super().__init__(c, data)
        self._pe_count = pe_count

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        src_node = super()._build_source(builder)
        src_node.launch_options.pe_count = self._pe_count
        return src_node


class SimpleStage(RecordThreadIdStage):

    def __init__(self, c: Config, pe_count: int):
        super().__init__(c)
        self._pe_count = pe_count

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = super()._build_single(builder, input_node)
        node.launch_options.pe_count = self._pe_count
        return node


@pytest.mark.parametrize("source_error, stage_error", [(True, False), (False, True), (True, True)])
def test_pe_exceeds_cores(source_error: bool, stage_error: bool):
    """
    Test to verify that when the pe_count exceeds the number of cores available, a reasonable error is raised.
    Issue #2202
    """
    config = Config()
    pipe = LinearPipeline(config)

    safe_pe_count = 1
    unsafe_pe_count = os.cpu_count() + 1

    if source_error:
        source_kwargs = {"pe_count": unsafe_pe_count}
    else:
        source_kwargs = {"pe_count": safe_pe_count}

    if stage_error:
        stage_kwargs = {"pe_count": unsafe_pe_count}
    else:
        stage_kwargs = {"pe_count": safe_pe_count}

    pipe.set_source(SimpleSource(config, [1, 2, 3], **source_kwargs))
    pipe.add_stage(SimpleStage(config, **stage_kwargs))

    with pytest.raises(RuntimeError, match="^more dedicated threads/cores than available$"):
        pipe.run()
