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

# When segment modules are imported, they're added to the module registry.
# To avoid flake8 warnings about unused code, the noqa flag is used during import.
import morpheus.modules  # noqa: F401
from morpheus.stages.general.multi_port_modules_stage import MultiPortModulesStage
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import MULTIPLEXER


@pytest.fixture(scope="function")
def unregistered_module_conf():
    unregistered_module_conf = {
        "module_id": "TestModule",
        "module_name": "test_multi_port_module",
        "namespace": "test_morpheus_modules",
        "input_ports": ["input_1", "input_2"],
        "output_ports": ["output_1", "output_2"]
    }
    yield unregistered_module_conf


@pytest.fixture(scope="function")
def registered_module_conf():
    registered_module_conf = {
        "module_id": MULTIPLEXER,
        "namespace": MORPHEUS_MODULE_NAMESPACE,
        "module_name": "multiplexer",
        "input_ports": ["input_1", "input_2"],
        "output_port": "output"
    }
    yield registered_module_conf


@pytest.mark.use_python
def test_constructor(config, unregistered_module_conf):

    mod_stage = MultiPortModulesStage(config,
                                      module_conf=unregistered_module_conf,
                                      input_ports_key="input_ports",
                                      output_ports_key="output_ports")

    assert mod_stage.name == "test_multi_port_module"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = mod_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0
    assert accepted_types[0] == typing.Any

    assert mod_stage.supports_cpp_node() is False


@pytest.mark.parametrize("input_ports,output_ports", [(["input_1", "input_2"], ["output"]), (["input_1"], ["output"])])
def test_with_unregistred_module(config, input_ports, output_ports, unregistered_module_conf):

    unregistered_module_conf["input_ports"] = input_ports
    unregistered_module_conf["output_ports"] = output_ports

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = MultiPortModulesStage(config,
                                      module_conf=unregistered_module_conf,
                                      input_ports_key="input_ports",
                                      output_ports_key="output_ports")

    # Raises error as test module doesn't exist in the namespace 'test_morpheus_modules'
    with pytest.raises(ValueError):
        mod_stage._build(mock_segment, mock_input_stream)


@pytest.mark.parametrize("input_ports,output_ports", [([], ["output"]), (["input"], []), ([], [])])
def test_empty_ports(config, unregistered_module_conf, input_ports, output_ports):

    unregistered_module_conf["input_ports"] = input_ports
    unregistered_module_conf["output_ports"] = output_ports

    with pytest.raises(ValueError):
        MultiPortModulesStage(config,
                              module_conf=unregistered_module_conf,
                              input_ports_key="input_ports",
                              output_ports_key="output_ports")


@pytest.mark.parametrize("input_ports,output_ports,expected_count",
                         [(["input_1", "input_2"], ["output"], 2), (["input_1"], ["output"], 1),
                          (["input_1", "input_2"], ["output_1", "output_2"], 2)])
def test_registered_module(config, registered_module_conf, input_ports, output_ports, expected_count):
    registered_module_conf["input_ports"] = input_ports
    registered_module_conf["output_ports"] = output_ports

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node.return_value = mock_node

    mod_stage = MultiPortModulesStage(config,
                                      module_conf=registered_module_conf,
                                      input_ports_key="input_ports",
                                      output_ports_key="output_ports")

    mod_stage._build(mock_segment, mock_input_stream)

    mock_segment.load_module.assert_called_once()
    assert mock_segment.make_edge.call_count == expected_count
