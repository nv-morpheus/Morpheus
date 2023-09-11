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


import os

import fsspec
import pytest

from _utils import TEST_DIRS
from morpheus.common import FileTypes
from morpheus.stages.input.file_source import FileSource


@pytest.fixture(name="input_file", scope="function")
def file_fixture():
    return fsspec.open(os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.json"))


@pytest.mark.use_python
def test_constructor(config):
    file_source = FileSource(
        config,
        files=["path/to/file.json", "path/to/another.json"],
        watch=True,
        sort_glob=True,
        recursive=False,
        queue_max_size=256,
        batch_timeout=10.0,
        file_type="json",
        repeat=3,
        filter_null=False,
        parser_kwargs={"key": "value"},
        watch_interval=2.0,
    )

    assert file_source._files == ["path/to/file.json", "path/to/another.json"]
    assert file_source._watch
    assert file_source._sort_glob
    assert not file_source._recursive
    assert file_source._queue_max_size == 256
    assert file_source._batch_timeout == 10.0
    assert file_source._file_type == "json"
    assert not file_source._filter_null
    assert file_source._parser_kwargs == {"key": "value"}
    assert file_source._watch_interval == 2.0
    assert file_source._repeat_count == 3


@pytest.mark.use_python
@pytest.mark.parametrize("input_files", [["file1.json", "file2.json"], []])
def test_constructor_with_invalid_params(config, input_files):
    with pytest.raises(ValueError):
        # 'watch' is True, but multiple files are provided
        FileSource(config, files=input_files, watch=True)


@pytest.mark.parametrize("input_files", [["file1.json", "file2.json"]])
def test_convert_to_fsspec_files(input_files):
    actual_output = FileSource.convert_to_fsspec_files(files=input_files)

    assert isinstance(actual_output, fsspec.core.OpenFiles)
    assert os.path.basename(actual_output[0].full_name) == input_files[0]
    assert os.path.basename(actual_output[1].full_name) == input_files[1]
