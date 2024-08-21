#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from _utils.stages.conv_msg import ConvMsg
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import ExecutionMode
from morpheus.pipeline.execution_mode_mixins import CpuOnlyMixin
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin


class CpuOnlyStage(CpuOnlyMixin, ConvMsg):
    pass


class GpuAndCpuStage(GpuAndCpuMixin, ConvMsg):
    pass


@pytest.mark.parametrize("stage_cls, expected_modes",
                         [(ConvMsg, {ExecutionMode.GPU}), (CpuOnlyStage, {ExecutionMode.CPU}),
                          (GpuAndCpuStage, {ExecutionMode.GPU, ExecutionMode.CPU})])
def test_execution_mode_mixins(stage_cls: type[ConvMsg], expected_modes: set):
    # intentionally not using the config fixture so that we can set the execution mode and avoid iterating over
    # python/C++ execution modes
    config = Config()
    if ExecutionMode.CPU in expected_modes:
        config.execution_mode = ExecutionMode.CPU
    else:
        config.execution_mode = ExecutionMode.GPU

    stage = stage_cls(config)
    assert set(stage.supported_execution_modes()) == expected_modes


@pytest.mark.parametrize("stage_cls", [ConvMsg, CpuOnlyStage])
def test_unsupported_mode_error(stage_cls: type[ConvMsg]):
    # intentionally not using the config fixture so that we can set the execution mode and avoid iterating over
    # python/C++ execution modes
    config = Config()
    if issubclass(stage_cls, CpuOnlyMixin):
        config.execution_mode = ExecutionMode.GPU
    else:
        config.execution_mode = ExecutionMode.CPU

    with pytest.raises(RuntimeError, match="Unsupported execution mode"):
        stage_cls(config)
