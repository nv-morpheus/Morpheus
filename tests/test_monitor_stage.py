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

import inspect
from unittest import mock

import pytest

import cudf

from morpheus.pipeline import general_stages as gs
from morpheus.pipeline.messages import MultiMessage


def test_constructor(config):
    # Intentionally not making assumptions about the defaults other than they exist
    # and still create a valid stage.
    m = gs.MonitorStage(config)
    assert m.name == "monitor"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = m.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    two_x = lambda x: x * 2
    m = gs.MonitorStage(config, description="Test Description", smoothing=0.7, unit='units', determine_count_fn=two_x)
    assert m._description == "Test Description"
    assert m._smoothing == 0.7
    assert m._unit == "units"
    assert m._determine_count_fn is two_x


@mock.patch('morpheus.pipeline.general_stages.MorpheusTqdm')
def test_on_start(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = gs.MonitorStage(config)
    assert m._progress is None

    m.on_start()
    mock_morph_tqdm.assert_called_once()
    mock_morph_tqdm.reset.assert_called_once()
    assert m._progress is mock_morph_tqdm


@mock.patch('morpheus.pipeline.general_stages.MorpheusTqdm')
def test_stop(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = gs.MonitorStage(config)
    assert m._progress is None

    # Calling on_stop is a noop if we are stopped
    m.stop()
    mock_morph_tqdm.assert_not_called()

    m.on_start()
    m.stop()
    mock_morph_tqdm.close.assert_called_once()


@mock.patch('morpheus.pipeline.general_stages.MorpheusTqdm')
def test_refresh(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = gs.MonitorStage(config)
    assert m._progress is None

    m.on_start()
    m._refresh_progress(None)
    mock_morph_tqdm.refresh.assert_called_once()


@mock.patch('morpheus.pipeline.general_stages.MorpheusTqdm')
def test_build_single(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    mock_stream = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_sink.return_value = mock_stream
    mock_input = mock.MagicMock()

    m = gs.MonitorStage(config)
    m._build_single(mock_segment, mock_input)
    m.on_start()

    mock_segment.make_sink.assert_called_once()
    mock_segment.make_edge.assert_called_once()

    sink_on_error = mock_segment.make_sink.call_args.args[2]
    sink_on_completed = mock_segment.make_sink.call_args.args[3]

    # This is currenlty just a log stmt, just verify that its callable
    sink_on_error(RuntimeError("unittest"))

    # Verify we close tqdm propperly on complete
    sink_on_completed()
    mock_morph_tqdm.stop.assert_called_once()


def test_auto_count_fn(config):
    m = gs.MonitorStage(config)

    assert m._auto_count_fn(None) is None
    assert m._auto_count_fn([]) is None

    # Ints not supported, lists are, but lists of unsupported are also unsupported
    pytest.raises(NotImplementedError, m._auto_count_fn, 1)
    pytest.raises(NotImplementedError, m._auto_count_fn, [1])

    # Just verify that we get a valid function for each supported type
    assert inspect.isfunction(m._auto_count_fn(['s']))
    assert inspect.isfunction(m._auto_count_fn('s'))
    assert inspect.isfunction(m._auto_count_fn(cudf.DataFrame()))
    assert inspect.isfunction(m._auto_count_fn(MultiMessage(None, 0, 0)))

    # Other iterables return the len function
    assert m._auto_count_fn({}) is len
    assert m._auto_count_fn(()) is len
    assert m._auto_count_fn(set()) is len


@mock.patch('morpheus.pipeline.general_stages.MorpheusTqdm')
def test_progress_sink(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = gs.MonitorStage(config)
    m.on_start()

    m._progress_sink(None)
    assert m._determine_count_fn is None
    mock_morph_tqdm.update.assert_not_called()

    m._progress_sink(MultiMessage(None, 0, 12))
    assert inspect.isfunction(m._determine_count_fn)
    mock_morph_tqdm.update.assert_called_once_with(n=12)
