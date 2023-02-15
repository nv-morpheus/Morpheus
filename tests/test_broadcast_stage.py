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

import pytest

from morpheus.stages.general.broadcast_stage import BroadcastStage


def test_constructor(config):

    b = BroadcastStage(config, output_port_count=3)
    assert b._output_port_count == 3

    # Just ensure that we get a valid non-empty tuple
    accepted_types = b.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    b = BroadcastStage(config)
    assert b._output_port_count == 2

    pytest.raises(AssertionError, BroadcastStage, config, output_port_count=0)


@pytest.mark.use_python
def test_build(config):
    mock_builder = mock.MagicMock()
    in_mock_stream_pairs = [(mock.MagicMock(), mock.MagicMock())]

    b = BroadcastStage(config)
    b._get_broadcast_node = mock.MagicMock()

    mock_out_stream_pairs = b._build(mock_builder, in_mock_stream_pairs)

    assert len(mock_out_stream_pairs) == 2

    mock_builder.make_edge.assert_called_once()
