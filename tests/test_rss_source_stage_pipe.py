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

from _utils import TEST_DIRS
from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.rss_source_stage import RSSSourceStage
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage

valid_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")
invalid_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xm")


@pytest.mark.use_python
def test_constructor_with_feed_url(config):

    url_feed_input = "https://realpython.com/atom.xml"
    rss_source_stage = RSSSourceStage(config, feed_input=url_feed_input)

    ctlr = rss_source_stage._controller

    assert ctlr._feed_input == {"https://realpython.com/atom.xml"}
    assert ctlr._run_indefinitely is True


@pytest.mark.use_python
def test_constructor_with_feed_file(config):
    file_feed_input = os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")
    rss_source_stage = RSSSourceStage(config,
                                      feed_input=file_feed_input,
                                      interval_secs=5,
                                      stop_after=10,
                                      max_retries=2,
                                      batch_size=256,
                                      enable_cache=True,
                                      cache_dir="./.cache/http_cache")

    ctlr = rss_source_stage._controller

    assert ctlr._feed_input == {file_feed_input}
    assert ctlr._run_indefinitely is False
    assert ctlr._batch_size == 256
    assert rss_source_stage._interval_secs == 5
    assert rss_source_stage._stop_after == 10
    assert rss_source_stage._max_retries == 2
    assert rss_source_stage._controller._session is not None
    assert rss_source_stage._controller._session.cache.cache_name == "./.cache/http_cache/RSSController.sqlite"


@pytest.mark.use_python
def test_support_cpp_node(config):
    url_feed_input = "https://realpython.com/atom.xml"
    rss_source_stage = RSSSourceStage(config, feed_input=url_feed_input)

    assert rss_source_stage.supports_cpp_node() is False


@pytest.mark.use_python
@pytest.mark.parametrize(
    "feed_input, batch_size, expected_count, enable_cache",
    [([valid_feed_input], 30, 1, False), ([valid_feed_input], 12, 3, True),
     ([valid_feed_input, valid_feed_input], 15, 2, False)
     # Duplicate feed inputs
     ])
def test_rss_source_stage_pipe(config: Config,
                               feed_input: list[str],
                               batch_size: int,
                               expected_count: int,
                               enable_cache: bool):

    pipe = Pipeline(config)

    rss_source_stage = pipe.add_stage(
        RSSSourceStage(config, feed_input=feed_input, batch_size=batch_size, enable_cache=enable_cache))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(rss_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


@pytest.mark.use_python
def test_invalid_input_rss_source_stage_pipe(config: Config):

    pipe = Pipeline(config)

    rss_source_stage = pipe.add_stage(
        RSSSourceStage(config, feed_input=[invalid_feed_input], interval_secs=1, max_retries=1))
    sink_stage = pipe.add_stage(InMemorySinkStage(config))

    pipe.add_edge(rss_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == 0
    assert rss_source_stage._controller._errored_feeds == [invalid_feed_input]
