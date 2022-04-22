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

from morpheus.pipeline.general_stages import AddClassificationsStage


def test_constructor(config):
    config.class_labels = ['frogs', 'lizards', 'toads']

    ac = AddClassificationsStage(config)
    assert ac._class_labels == ['frogs', 'lizards', 'toads']
    assert ac._labels == ['frogs', 'lizards', 'toads']
    assert ac._idx2label == {0: 'frogs', 1: 'lizards', 2: 'toads'}
    assert ac.name == "add-class"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = ac.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    ac = AddClassificationsStage(config, threshold=1.3, labels=['lizards'], prefix='test_')
    assert ac._class_labels, ['frogs', 'lizards', 'toads']
    assert ac._labels, ['lizards']
    assert ac._idx2label, {1: 'test_lizards'}

    pytest.raises(AssertionError, AddClassificationsStage, config, labels=['missing'])


@pytest.mark.use_python
def test_add_labels(config):
    mock_message = mock.MagicMock()
    mock_message.probs = cp.array([[0.1, 0.5, 0.8], [0.2, 0.6, 0.9]])

    config.class_labels = ['frogs', 'lizards', 'toads']

    ac = AddClassificationsStage(config, threshold=0.5)
    ac._add_labels(mock_message)

    mock_message.set_meta.assert_has_calls([
        mock.call('frogs', [False, False]),
        mock.call('lizards', [False, True]),
        mock.call('toads', [True, True]),
    ])

    wrong_shape = mock.MagicMock()
    wrong_shape.probs = cp.array([[0.1, 0.5], [0.2, 0.6]])
    pytest.raises(RuntimeError, ac._add_labels, wrong_shape)


@pytest.mark.use_python
def test_build_single(config):
    mock_stream = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node.return_value = mock_stream
    mock_input = mock.MagicMock()

    config.class_labels = ['frogs', 'lizards', 'toads']

    ac = AddClassificationsStage(config)
    ac._build_single(mock_segment, mock_input)

    mock_segment.make_node.assert_called_once()
    mock_segment.make_edge.assert_called_once()
