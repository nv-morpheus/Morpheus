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
import typing_utils

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.pipeline import Pipeline
from morpheus.pipeline.base_stage import BaseStage
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stage_schema import PortSchema
from morpheus.pipeline.stage_schema import StageSchema
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from _utils.dataset_manager import DatasetManager
from _utils.stages.split_stage import SplitStage


@pytest.fixture(name="in_mem_source_stage")
def in_mem_source_stage_fixture(config: Config, dataset_cudf: DatasetManager):
    df = dataset_cudf['filter_probs.csv']
    yield InMemorySourceStage(config, [df])


def _build_ports(config: Config, source_stage: SourceStage, stage: BaseStage):
    pipe = Pipeline(config)
    pipe.add_stage(source_stage)
    pipe.add_stage(stage)
    pipe.add_edge(source_stage, stage)
    pipe.build()


@pytest.fixture(name="stage")
def stage_fixture(config: Config, in_mem_source_stage: InMemorySourceStage):
    stage = DeserializeStage(config)
    _build_ports(config=config, source_stage=in_mem_source_stage, stage=stage)

    yield stage


@pytest.fixture(name="multiport_stage")
def multiport_stage_fixture(config: Config, in_mem_source_stage: InMemorySourceStage):
    stage = SplitStage(config)
    _build_ports(config=config, source_stage=in_mem_source_stage, stage=stage)

    yield stage


@pytest.mark.parametrize("stage_fixture_name,num_inputs,num_outputs", [("in_mem_source_stage", 0, 1), ("stage", 1, 1),
                                                                       ("multiport_stage", 1, 2)])
def test_constructor(request: pytest.FixtureRequest, stage_fixture_name: str, num_inputs: int, num_outputs: int):
    stage = request.getfixturevalue(stage_fixture_name)
    schema = StageSchema(stage)
    assert len(schema.input_schemas) == num_inputs
    assert len(schema.output_schemas) == num_outputs


def test_single_port_input_schemas(stage: DeserializeStage):
    schema = StageSchema(stage)
    assert len(schema.input_schemas) == 1

    port_schema = schema.input_schemas[0]
    assert port_schema.get_type() is MessageMeta

    assert schema.input_schema is port_schema


def test_single_port_input_types(stage: DeserializeStage):
    schema = StageSchema(stage)
    assert len(schema.input_types) == 1

    assert schema.input_types[0] is MessageMeta
    assert schema.input_type is MessageMeta


def test_single_port_output_schemas(in_mem_source_stage: InMemorySourceStage):
    schema = StageSchema(in_mem_source_stage)
    in_mem_source_stage.compute_schema(schema)
    assert len(schema.output_schemas) == 1

    port_schema = schema.output_schemas[0]
    assert port_schema.get_type() is MessageMeta

    assert schema.output_schema is port_schema


def test_multi_port_output_schemas(multiport_stage: SplitStage):
    schema = StageSchema(multiport_stage)
    multiport_stage.compute_schema(schema)
    assert len(schema.output_schemas) == 2

    for port_schema in schema.output_schemas:
        assert port_schema.get_type() is MessageMeta


def test_output_schema_multi_error(multiport_stage: SplitStage):
    """
    Test confirms that the output_schema property raises an error when there are multiple output schemas
    """
    schema = StageSchema(multiport_stage)
    multiport_stage.compute_schema(schema)
    assert len(schema.output_schemas) == 2

    with pytest.raises(AssertionError):
        schema.output_schema
