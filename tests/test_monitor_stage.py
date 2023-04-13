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
import threading
import typing
from unittest import mock

import mrc
import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.single_port_stage import SinglePortStage
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
@mock.patch('mrc.Builder.make_node_component')
@mock.patch('mrc.Builder.make_edge')
def test_log_level(mock_make_edge, mock_make_node_component, config, morpheus_log_level):
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
    assert mock_make_node_component.call_count == expected_call_count


@pytest.mark.usefixtures("reset_loglevel")
@pytest.mark.use_python
def test_thread(config):
    """
    Test ensures the monitor stage doesn't add itself to the MRC pipeline if not configured for the current log-level
    """
    input_file = os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.csv")

    set_log_level(log_level=logging.INFO)

    # Create a dummy forwarding stage that allows us to save the thread id from this progress engine
    class DummyStage(SinglePortStage):

        def __init__(self, c: Config):
            super().__init__(c)

            self.thread_id = None

        @property
        def name(self):
            return "dummy"

        def accepted_types(self):
            return (typing.Any, )

        def supports_cpp_node(self):
            return False

        def _save_thread(self, x):
            self.thread_id = threading.current_thread().ident
            return x

        def _build_single(self, builder: mrc.Builder, input_stream):
            stream = builder.make_node(self.unique_name, mrc.core.operators.map(self._save_thread))

            builder.make_edge(input_stream[0], stream)

            return stream, input_stream[1]

    monitor_thread_id = None

    # Create a dummy count function where we can save the thread id from the monitor stage
    def fake_determine_count_fn(x):
        nonlocal monitor_thread_id

        monitor_thread_id = threading.current_thread().ident

        return x.count

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=input_file))
    dummy_stage = pipe.add_stage(DummyStage(config))
    pipe.add_stage(MonitorStage(config, determine_count_fn=fake_determine_count_fn))
    pipe.run()

    # Check that the thread ids are the same
    assert dummy_stage.thread_id == monitor_thread_id
