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

import os

import pytest

import cudf

from morpheus.utils.controllers.rss_controller import RSSController
from utils import TEST_DIRS

test_urls = ["https://realpython.com/atom.xml", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]

test_invalid_urls = [
    "invalid_url",
    "example.com/rss-feed-url",
    "ftp://",
]

test_file_paths = [os.path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")]

test_invalid_file_paths = [
    "/path/to/nonexistent_file.xml",
    "/path/to/directory/",
]


@pytest.mark.parametrize("feed_input, expected_output", [(url, True) for url in test_urls])
def test_run_indefinitely_true(feed_input, expected_output):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely == expected_output


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths + test_file_paths)
def test_run_indefinitely_false(feed_input):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely is False


@pytest.mark.parametrize("feed_input", test_urls)
def test_parse_feed_valid_url(feed_input):
    controller = RSSController(feed_input=feed_input)
    feed = controller.parse_feed()
    assert feed.entries


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths)
def test_parse_feed_invalid_input(feed_input):
    controller = RSSController(feed_input=feed_input)
    with pytest.raises(Exception):
        controller.parse_feed()


@pytest.mark.parametrize("feed_input", test_urls + test_file_paths)
def test_fetch_message_metas(feed_input):
    controller = RSSController(feed_input=feed_input)
    dataframes_generator = controller.fetch_dataframes()
    dataframe = next(dataframes_generator, None)
    assert isinstance(dataframe, cudf.DataFrame)
    assert len(dataframe) > 0


@pytest.mark.parametrize("feed_input", test_file_paths)
def test_create_dataframe(feed_input):
    controller = RSSController(feed_input=feed_input)
    entries = [{'id': '1', 'title': 'Entry 1'}, {'id': '2', 'title': 'Entry 2'}]
    df = controller.create_dataframe(entries)
    assert len(df) == len(entries)


@pytest.mark.parametrize("feed_input", test_urls)
def test_is_url_true(feed_input):
    assert RSSController.is_url(feed_input)


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths + test_file_paths)
def test_is_url_false(feed_input):
    assert not RSSController.is_url(feed_input)
