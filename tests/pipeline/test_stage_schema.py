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

import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline import Pipeline
from morpheus.pipeline.base_stage import BaseStage
from morpheus.pipeline.stage_schema import PortSchema
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from _utils.dataset_manager import DatasetManager
from _utils.stages.split_stage import SplitStage


@pytest.fixture(name="source_stage")
def source_stage_fixture(config: Config, dataset_cudf: DatasetManager):
    df = dataset_cudf['filter_probs.csv']
    yield InMemorySourceStage(config, [df])


@pytest.fixture(name="stage")
def stage_fixture(config: Config, source_stage: InMemorySourceStage):
    stage = DeserializeStage(config)

    pipe = LinearPipeline(config)
    pipe.set_source(source_stage)
    pipe.add_stage(stage)
    pipe.build()

    yield stage


@pytest.fixture(name="multiport_stage")
def multiport_stage_fixture(config: Config, source_stage: InMemorySourceStage):
    stage = SplitStage(config)

    pipe = Pipeline(config)
    pipe.add_stage(source_stage)
    pipe.add_stage(stage)
    pipe.add_edge(source_stage, stage)
    pipe.build()

    yield stage


@pytest.mark.parametrize("stage_fixture,num_inputs,num_outputs", [("source_stage", 0, 1), ("stage", 1, 1),
                                                                  ("multiport_stage", 1, 2)])
def test_constructor(request: pytest.FixtureRequest, stage_fixture: str, num_inputs: int, num_outputs: int):
    stage = request.getfixturevalue(stage_fixture)
    schema = StageSchema(stage)
    assert len(schema.input_schemas) == num_inputs
    assert len(schema.output_schemas) == num_outputs
