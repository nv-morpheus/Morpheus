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

import os

import pytest

from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage
from utils import TEST_DIRS


@pytest.mark.use_python
def test_constructor(config):

    url_feed_input = "https://realpython.com/atom.xml"
    rss_source_stage = RSSSourceStage(config, feed_input=url_feed_input)

    file_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")
    rss_source_stage_2 = RSSSourceStage(config,
                                        feed_input=file_feed_input,
                                        interval_secs=5,
                                        stop_after=10,
                                        max_retries=2)

    ctlr = rss_source_stage._controller
    ctlr_2 = rss_source_stage_2._controller

    assert ctlr._feed_input == "https://realpython.com/atom.xml"
    assert ctlr._run_indefinitely is True
    assert ctlr._batch_size == config.pipeline_batch_size
    assert rss_source_stage._interval_secs == 600
    assert rss_source_stage._stop_after == 0
    assert rss_source_stage._max_retries == 5

    assert ctlr_2._feed_input == file_feed_input
    assert ctlr_2._run_indefinitely is False
    assert ctlr_2._batch_size == config.pipeline_batch_size
    assert rss_source_stage_2._interval_secs == 5
    assert rss_source_stage_2._stop_after == 10
    assert rss_source_stage_2._max_retries == 2

    assert rss_source_stage.supports_cpp_node() is False
    assert rss_source_stage_2.supports_cpp_node() is False


@pytest.mark.use_python
@pytest.mark.parametrize("batch_size, expected_count", [(30, 1), (12, 3), (15, 2)])
def test_rss_source_stage_pipe(config, batch_size, expected_count) -> None:

    feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")
    config.pipeline_batch_size = batch_size

    pipe = Pipeline(config)

    rss_source_stage = pipe.add_stage(RSSSourceStage(config, feed_input=feed_input))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(rss_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


@pytest.mark.use_python
def test_invalid_input_rss_source_stage_pipe(config) -> None:

    feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xm")

    pipe = Pipeline(config)

    rss_source_stage = pipe.add_stage(RSSSourceStage(config, feed_input=feed_input, interval_secs=1, max_retries=1))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(rss_source_stage, sink_stage)

    with pytest.raises(Exception):
        pipe.run()
