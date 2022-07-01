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

import cupy as cp
import pytest

from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage


def test_constructor(config):
    fds = FilterDetectionsStage(config)
    assert fds.name == "filter"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = fds.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    fds = FilterDetectionsStage(config, threshold=0.2)
    assert fds._threshold == 0.2


@pytest.mark.use_python
def test_filter(config):
    fds = FilterDetectionsStage(config, threshold=0.5)

    def make_mock_message(probs_, offset=8):
        count = len(probs_)
        mm = mock.MagicMock()
        mm.mess_offset = offset
        mm.probs = probs_
        mm.mess_count = count
        mm.meta.count = count
        mm.mask = cp.ones(count, cp.bool_)
        return mm

    probs = cp.array([[0.1, 0.5, 0.3], [0.2, 0.3, 0.4]])
    mock_message = make_mock_message(probs)

    # All values are at or below the threshold
    output_message = fds.filter(mock_message)
    assert output_message.mask.tolist() == [False, False]

    # Only one row has a value above the threshold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.5, 0.8],
        [0.2, 0.4, 0.3],
    ])

    mock_message = make_mock_message(probs)

    output_message = fds.filter(mock_message)
    assert output_message.mask.tolist() == [False, True, False]

    # Two adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = make_mock_message(probs)

    output_message = fds.filter(mock_message)
    assert output_message.mask.tolist() == [False, False, True, True, False]

    # Two non-adjacent rows have a value above the threashold
    probs = cp.array([
        [0.2, 0.4, 0.3],
        [0.1, 0.2, 0.3],
        [0.1, 0.5, 0.8],
        [0.4, 0.3, 0.2],
        [0.1, 0.9, 0.2],
        [0.2, 0.4, 0.3],
    ])

    mock_message = make_mock_message(probs)

    output_message = fds.filter(mock_message)
    assert output_message.mask.tolist() == [False, False, True, False, True, False]


@pytest.mark.use_python
def test_build_single(config):
    mock_stream = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_builder.make_node.return_value = mock_stream
    mock_input = mock.MagicMock()

    fds = FilterDetectionsStage(config)
    fds._build_single(mock_builder, mock_input)

    mock_builder.make_node.assert_called_once()
    mock_builder.make_edge.assert_called_once()
