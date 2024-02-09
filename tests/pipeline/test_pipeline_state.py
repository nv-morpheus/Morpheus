#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.pipeline import PipelineState
from morpheus.pipeline.stage_decorator import source


@source
def source_test_stage() -> int:
    for i in range(10):
        yield i

# pylint: disable=too-many-function-args


def test_normal_run(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.set_source(source_test_stage(config))
    pipeline.run()
    assert pipeline.state == PipelineState.COMPLETED


async def test_normal_build_and_start(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED


async def test_stop_after_start(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    # pipeline.set_source(TestSourceStage(config))
    pipeline.set_source(source_test_stage(config))
    await pipeline.build_and_start()
    assert pipeline.state == PipelineState.STARTED
    pipeline.stop()
    assert pipeline.state == PipelineState.STOPPED
    await pipeline.join()
    assert pipeline.state == PipelineState.COMPLETED


def test_stop_after_run(config: Config):
    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.set_source(source_test_stage(config))
    pipeline.run()
    assert pipeline.state == PipelineState.COMPLETED
    with pytest.raises(Exception):
        pipeline.stop()


def test_stop_without_start(config: Config):

    pipeline = LinearPipeline(config)
    assert pipeline.state == PipelineState.INITIALIZED
    # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipeline.set_source(source_test_stage(config))
    with pytest.raises(Exception):
        pipeline.stop()


# async def test_stop_after_stop(config: Config):

#     pipeline = LinearPipeline(config)
#     assert pipeline.state == PipelineState.INITIALIZED
#     # pipeline.set_source(InMemorySourceStage(config, [filter_probs_df]))
#     pipeline.set_source(source_test_stage(config))
#     await pipeline.build_and_start()
#     assert pipeline.state == PipelineState.STARTED
#     pipeline.stop()
#     assert pipeline.state == PipelineState.STOPPED
#     with pytest.raises(Exception):
#         pipeline.stop()
