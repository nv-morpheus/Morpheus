#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
from unittest import mock

import pytest

import cudf

from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import set_log_level
from utils import TEST_DIRS


def test_constructor(config):
    # Intentionally not making assumptions about the defaults other than they exist
    # and still create a valid stage.
    m = MonitorStage(config, log_level=logging.WARNING)
    assert m.name == "monitor"

    # Just ensure that we get a valid non-empty tuple
    accepted_types = m.accepted_types()
    assert isinstance(accepted_types, tuple)
    assert len(accepted_types) > 0

    def two_x(x):
        return x * 2

    m = MonitorStage(config, description="Test Description", smoothing=0.7, unit='units', determine_count_fn=two_x)
    assert m._description == "Test Description"
    assert m._smoothing == 0.7
    assert m._unit == "units"
    assert m._determine_count_fn is two_x


@mock.patch('morpheus.stages.general.monitor_stage.MorpheusTqdm')
def test_on_start(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = MonitorStage(config, log_level=logging.WARNING)
    assert m._progress is None

    m.on_start()
    mock_morph_tqdm.assert_called_once()
    mock_morph_tqdm.reset.assert_called_once()
    assert m._progress is mock_morph_tqdm


@mock.patch('morpheus.stages.general.monitor_stage.MorpheusTqdm')
def test_stop(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = MonitorStage(config, log_level=logging.WARNING)
    assert m._progress is None

    # Calling on_stop is a noop if we are stopped
    m.stop()
    mock_morph_tqdm.assert_not_called()

    m.on_start()
    m.stop()
    mock_morph_tqdm.close.assert_called_once()


@mock.patch('morpheus.stages.general.monitor_stage.MorpheusTqdm')
def test_refresh(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = MonitorStage(config, log_level=logging.WARNING)
    assert m._progress is None

    m.on_start()
    m._refresh_progress(None)
    mock_morph_tqdm.refresh.assert_called_once()


@mock.patch('morpheus.stages.general.monitor_stage.ops')
@mock.patch('morpheus.stages.general.monitor_stage.MorpheusTqdm')
def test_build_single(mock_morph_tqdm, mock_operators, config):
    MonitorStage.stage_count = 0
    mock_morph_tqdm.return_value = mock_morph_tqdm
    mock_morph_tqdm.monitor = mock.MagicMock()

    mock_stream = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_node_full.return_value = mock_stream
    mock_input = mock.MagicMock()

    m = MonitorStage(config, log_level=logging.WARNING)
    m._build_single(mock_segment, mock_input)
    m.on_start()

    assert MonitorStage.stage_count == 1

    mock_segment.make_node_full.assert_called_once()
    mock_segment.make_edge.assert_called_once()

    node_fn = mock_segment.make_node_full.call_args.args[1]

    mock_observable = mock.MagicMock()
    mock_subscriber = mock.MagicMock()

    node_fn(mock_observable, mock_subscriber)
    mock_operators.on_completed.assert_called_once()
    sink_on_completed = mock_operators.on_completed.call_args.args[0]

    # Verify we close tqdm properly on complete
    sink_on_completed()
    mock_morph_tqdm.stop.assert_called_once()


def test_auto_count_fn(config):
    m = MonitorStage(config, log_level=logging.WARNING)

    assert m._auto_count_fn(None) is None
    assert m._auto_count_fn([]) is None

    # Ints not supported, lists are, but lists of unsupported are also unsupported
    pytest.raises(NotImplementedError, m._auto_count_fn, 1)
    pytest.raises(NotImplementedError, m._auto_count_fn, [1])

    # Just verify that we get a valid function for each supported type
    assert inspect.isfunction(m._auto_count_fn(['s']))
    assert inspect.isfunction(m._auto_count_fn('s'))
    assert inspect.isfunction(m._auto_count_fn(cudf.DataFrame()))
    assert inspect.isfunction(
        m._auto_count_fn(MultiMessage(meta=MessageMeta(df=cudf.DataFrame(range(12), columns=["test"])))))

    # Other iterables return the len function
    assert m._auto_count_fn({}) is len
    assert m._auto_count_fn(()) is len
    assert m._auto_count_fn(set()) is len


@mock.patch('morpheus.stages.general.monitor_stage.MorpheusTqdm')
def test_progress_sink(mock_morph_tqdm, config):
    mock_morph_tqdm.return_value = mock_morph_tqdm

    m = MonitorStage(config, log_level=logging.WARNING)
    m.on_start()

    m._progress_sink(None)
    assert m._determine_count_fn is None
    mock_morph_tqdm.update.assert_not_called()

    m._progress_sink(MultiMessage(meta=MessageMeta(df=cudf.DataFrame(range(12), columns=["test"]))))
    assert inspect.isfunction(m._determine_count_fn)
    mock_morph_tqdm.update.assert_called_once_with(n=12)


@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.parametrize('morpheus_log_level',
                         [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG])
@mock.patch('mrc.Builder.make_node_full')
@mock.patch('mrc.Builder.make_edge')
def test_log_level(mock_make_edge, mock_make_node_full, config, morpheus_log_level):
    """
    Test ensures the monitor stage doesn't add itself to the MRC pipeline if not configured for the current log-level
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    set_log_level(morpheus_log_level)
    monitor_stage_level = logging.INFO

    should_be_included = (morpheus_log_level <= monitor_stage_level)

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))

    ms = MonitorStage(config, log_level=monitor_stage_level)

    pipe.add_stage(ms)
    pipe.run()

    expected_call_count = 1 if should_be_included else 0
    assert mock_make_node_full.call_count == expected_call_count
