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

from unittest import mock

from morpheus.config import Config
from morpheus.llm import LLMEngine
from morpheus.messages import ControlMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.stages.llm.llm_engine_stage import LLMEngineStage


def test_constructor(config: Config):
    stage = LLMEngineStage(config, engine=mock.MagicMock(LLMEngine))
    assert isinstance(stage, SinglePortStage)


def test_name(config: Config):
    stage = LLMEngineStage(config, engine=mock.MagicMock(LLMEngine))
    assert isinstance(stage.name, str)
    assert len(stage.name) > 0


def test_accepted_types(config: Config):
    stage = LLMEngineStage(config, engine=mock.MagicMock(LLMEngine))
    assert stage.accepted_types() == (ControlMessage, )


def test_supports_cpp_node(config: Config):
    stage = LLMEngineStage(config, engine=mock.MagicMock(LLMEngine))
    assert stage.supports_cpp_node()
