#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import srf

from morpheus.stages.general.module_stage import ModuleStage

module_config = {
    "module_id": "dfp_training",
    "module_name": "DFPTrainingModule",
    "module_namespace": "test",
    "version": [22, 11, 0],
    "input_type_class": "morpheus.messages.multi_dfp_message.MultiDFPMessage",
    "output_type_class": "morpheus.messages.multi_ae_message.MultiAEMessage"
}


def test_constructor(config):

    mod_stage = ModuleStage(config, module_config)
    assert mod_stage._module_id == "dfp_training"
    assert mod_stage._module_name == "DFPTrainingModule"
    assert mod_stage.name == "DFPTrainingModule"
    assert mod_stage._module_ns == "test"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = mod_stage.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    pytest.raises(NotImplementedError, mod_stage._get_cpp_module_node, None)


@pytest.mark.use_python
def test_build_single(config):
    
    registry = srf.ModuleRegistry()

    mock_node = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_module = mock.MagicMock()
    mock_input_stream = mock.MagicMock()

    mock_segment.load_module.return_value = mock_module
    mock_segment.make_node_full.return_value = mock_node

    mod_stage = ModuleStage(config, module_config)
    mod_stage._build_single(mock_segment, mock_input_stream)

    mock_segment.load_module.assert_called_once()
    mock_segment.make_edge.assert_called_once()

    expected = registry.registered_modules()["test"][0]
    acutal = "dfp_training"

    assert acutal == expected

    