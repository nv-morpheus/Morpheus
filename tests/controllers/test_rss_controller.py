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

from os import path

import feedparser
import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.controllers.rss_controller import RSSController

test_urls = ["https://realpython.com/atom.xml", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]

test_invalid_urls = [
    "invalid_url",
    "example.com/rss-feed-url",
    "ftp://",
]

test_file_paths = [path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml")]

test_invalid_file_paths = [
    "/path/to/nonexistent_file.xml",
    "/path/to/directory/",
]


@pytest.mark.parametrize("feed_input, expected_output", [(url, True) for url in test_urls])
def test_run_indefinitely_true(feed_input: str, expected_output: bool):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely == expected_output


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths + test_file_paths)
def test_run_indefinitely_false(feed_input: str):
    controller = RSSController(feed_input=feed_input)
    assert controller.run_indefinitely is False


@pytest.mark.parametrize("feed_input", test_urls)
def test_parse_feed_valid_url(feed_input: str):
    controller = RSSController(feed_input=feed_input)
    feed = list(controller.parse_feeds())[0]
    assert feed.entries


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths)
def test_parse_feed_invalid_input(feed_input: str):
    controller = RSSController(feed_input=feed_input)
    list(controller.parse_feeds())
    assert controller._errored_feeds == [feed_input]


@pytest.mark.parametrize("feed_input", [(test_urls + test_file_paths), test_urls, test_urls[0], test_file_paths[0]])
def test_fetch_dataframes(feed_input: str | list[str]):
    controller = RSSController(feed_input=feed_input)
    dataframes_generator = controller.fetch_dataframes()
    dataframe = next(dataframes_generator, None)
    assert isinstance(dataframe, cudf.DataFrame)
    assert "link" in dataframe.columns
    assert len(dataframe) > 0

@pytest.mark.parametrize("feed_input, expected_count", [(path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml"), 30)])
def test_skip_duplicates_feed_inputs(feed_input: str, expected_count: int):
    controller = RSSController(feed_input=[feed_input, feed_input])  # Pass duplicate feed inputs
    dataframes_generator = controller.fetch_dataframes()
    dataframe = next(dataframes_generator, None)
    assert isinstance(dataframe, cudf.DataFrame)
    assert len(dataframe) == expected_count

@pytest.mark.parametrize("feed_input", test_file_paths)
def test_create_dataframe(feed_input: str):
    controller = RSSController(feed_input=feed_input)
    entries = [{"id": "1", "title": "Entry 1"}, {"id": "2", "title": "Entry 2"}]
    df = controller.create_dataframe(entries)
    assert len(df) == len(entries)


@pytest.mark.parametrize("feed_input", test_urls)
def test_is_url_true(feed_input: str):
    assert RSSController.is_url(feed_input)


@pytest.mark.parametrize("feed_input", test_invalid_urls + test_invalid_file_paths + test_file_paths)
def test_is_url_false(feed_input: str):
    assert not RSSController.is_url(feed_input)


@pytest.mark.parametrize("feed_input, batch_size", [(test_urls + test_file_paths, 5)])
def test_batch_size(feed_input: str | list[str], batch_size: int):
    controller = RSSController(feed_input=feed_input, batch_size=batch_size)
    for df in controller.fetch_dataframes():
        assert isinstance(df, cudf.DataFrame)
        assert len(df) <= batch_size


@pytest.mark.parametrize("feed_input, is_url", [(path.join(TEST_DIRS.tests_data_dir, "rss_feed_atom.xml"), False),
                                                ("https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", True),
                                                ("https://www.mandiant.com/resources/blog/rss.xml", True)])
def test_try_parse_feed_with_beautiful_soup(feed_input: str, is_url: bool):
    rss_controller = RSSController(feed_input=feed_input)

    feed_data = rss_controller._try_parse_feed_with_beautiful_soup(feed_input, is_url)

    assert isinstance(feed_data, feedparser.FeedParserDict)

    assert len(feed_data.entries) > 0

    for entry in feed_data.entries:
        assert "title" in entry
        assert "link" in entry
        assert "id" in entry

        # Add more assertions as needed to validate the content of each entry
        for key, value in entry.items():
            if key not in ["title", "link", "id"]:
                assert value is not None

    # Additional assertions to validate the overall structure of the feed data
    assert isinstance(feed_data, dict)
    assert "entries" in feed_data
    assert isinstance(feed_data["entries"], list)
