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

import os
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.stages.input.arxiv_source import ArxivSource


def test_constructor(config: Config):
    cache_dir = "/does/not/exist"
    stage = ArxivSource(config, query="unittest", cache_dir=cache_dir)
    assert stage._query == "unittest"
    assert stage._cache_dir == cache_dir

    assert not os.path.exists(cache_dir)


def _make_mock_result(file_name: str):
    result = mock.MagicMock()
    result._get_default_filename.return_value = file_name
    return result


@pytest.mark.parametrize("use_subdir", [False, True])
@mock.patch("arxiv.Search")
def test_generate_frames_cache_miss(mock_search: mock.MagicMock, config: Config, tmp_path: str, use_subdir: bool):
    if use_subdir:
        # Tests that the cache directory is created if it doesn't exist
        cache_dir = os.path.join(tmp_path, "cache")
        assert not os.path.exists(cache_dir)
    else:
        cache_dir = tmp_path

    mock_search.return_value = mock_search
    mock_search.results.return_value = [_make_mock_result("apples.pdf"), _make_mock_result("plums.pdf")]
    stage = ArxivSource(config, query="unittest", cache_dir=cache_dir)

    expected_file_paths = [os.path.join(cache_dir, "apples.pdf"), os.path.join(cache_dir, "plums.pdf")]
    assert list(stage._generate_frames()) == expected_file_paths

    assert os.path.exists(cache_dir)


def test_generate_frames_cache_hit(tmp_path: str):
    pass
