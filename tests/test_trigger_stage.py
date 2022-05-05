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

from morpheus.stages.general.general_stages import TriggerStage


def test_constructor(config):
    ts = TriggerStage(config)
    assert ts.name == "trigger"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = ts.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0


@pytest.mark.use_python
def test_build_single(config):
    mock_stream = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node.return_value = mock_stream
    mock_input = mock.MagicMock()

    ts = TriggerStage(config)
    ts._build_single(mock_segment, mock_input)

    mock_segment.make_node_full.assert_called_once()
    mock_segment.make_edge.assert_called_once()
