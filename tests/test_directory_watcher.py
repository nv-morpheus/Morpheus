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

import os
from unittest import mock

import pytest

from morpheus.pipeline.input.directory_watcher import DirectoryWatcher
from utils import TEST_DIRS


@pytest.mark.use_python
@pytest.mark.parametrize('watch_directory', [False])
@pytest.mark.parametrize('max_files', [-1])
@pytest.mark.parametrize('sort_glob', [False])
def test_build_node(watch_directory, max_files, sort_glob):
    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', '*', '*.json')
    watcher = DirectoryWatcher(input_glob, watch_directory, max_files, sort_glob)

    assert watcher._watch_directory == False
    assert watcher._max_files == -1
    assert watcher._sort_glob == False

    mock_files = mock.MagicMock()
    mock_segment = mock.MagicMock()
    mock_segment.make_source.return_value = mock_files

    watcher.build_node('watch_directory', mock_segment)
    mock_segment.make_source.assert_called_once()
