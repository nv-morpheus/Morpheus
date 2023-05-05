#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing
from unittest import mock

import pytest

from morpheus.stages.general.multi_inputport_modules_stage import MultiInputportModulesStage

module_conf = {
    "module_id": "TestModule", "module_name": "test_multi_inputport_module", "namespace": "test_morpheus_modules"
}


@pytest.mark.use_python
def test_constructor(config):

    mod_stage = MultiInputportModulesStage(config, module_conf=module_conf, num_input_ports_to_merge=2)

    assert mod_stage.name == "test_multi_inputport_module"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = mod_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0
    assert accepted_types[0] == typing.Any

    assert mod_stage.supports_cpp_node() is False


@pytest.mark.use_python
def test_build_single_before_module_registration(config):

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = MultiInputportModulesStage(config, module_conf=module_conf, num_input_ports_to_merge=2)

    with pytest.raises(Exception):
        mod_stage._build(mock_segment, mock_input_stream)


@pytest.mark.use_python
def test_build_after_module_registration(config, register_test_module):

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = MultiInputportModulesStage(config, module_conf=module_conf, num_input_ports_to_merge=2)

    mod_stage._build(mock_segment, mock_input_stream)

    mock_segment.load_module.assert_called_once()
    assert mock_segment.make_edge.call_count == 2


@pytest.mark.use_python
def test_invalid_number_input_ports(config, register_test_module):
    num_input_ports_to_merge = 0
    with pytest.raises(ValueError):
        MultiInputportModulesStage(config, module_conf=module_conf, num_input_ports_to_merge=num_input_ports_to_merge)
