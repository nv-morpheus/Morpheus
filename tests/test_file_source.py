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

import cudf

from _utils import TEST_DIRS
from morpheus.common import FileTypes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.file_source import FileSource
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


@pytest.fixture(name="files", scope="function")
def files_fixture():

    return fsspec.open(os.path.join(TEST_DIRS.tests_data_dir, "filter_probs.json"))


@pytest.mark.use_python
def test_constructor(config):
    file_source = FileSource(
        config,
        files=["path/to/*.json"],
        watch=True,
        sort_glob=True,
        recursive=False,
        queue_max_size=256,
        batch_timeout=10.0,
        file_type=FileTypes.JSON,
        repeat=3,
        filter_null=False,
        parser_kwargs={"key": "value"},
        watch_interval=2.0,
    )

    assert file_source._files == ["path/to/*.json"]
    assert file_source._watch
    assert file_source._sort_glob
    assert not file_source._recursive
    assert file_source._queue_max_size == 256
    assert file_source._batch_timeout == 10.0
    assert file_source._file_type == FileTypes.JSON
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


@pytest.mark.use_python
@pytest.mark.parametrize(
    "input_file,filetypes,filter_null,parser_kwargs, repeat_count, expected_count, expected_df_count",
    [("filter_probs.json", FileTypes.Auto, False, {
        "lines": False
    }, 1, 1, 20), ("filter_probs.csv", FileTypes.CSV, False, {}, 2, 2, 20),
     ("filter_probs.jsonlines", FileTypes.JSON, False, {
         "lines": True
     }, 1, 1, 20)])
def test_generate_frames(input_file,
                         filetypes,
                         filter_null,
                         parser_kwargs,
                         repeat_count,
                         expected_count,
                         expected_df_count):
    in_file = fsspec.open(os.path.join(TEST_DIRS.tests_data_dir, input_file))

    metas = FileSource.generate_frames(file=in_file,
                                       file_type=filetypes,
                                       filter_null=filter_null,
                                       parser_kwargs=parser_kwargs,
                                       repeat_count=repeat_count)

    assert len(metas) == expected_count
    assert len(metas[0].df.columns) == 4
    assert len(metas[0].df) == expected_df_count
    assert isinstance(metas[0], MessageMeta)
    assert isinstance(metas[0].df, cudf.DataFrame)


@pytest.mark.use_python
@pytest.mark.parametrize("input_files,parser_kwargs,repeat,expected_count",
                         [([
                             "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T00_05_06.806Z.json",
                             "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T12_09_47.901Z.json"
                         ], {
                             "lines": False, "orient": "records"
                         },
                           1,
                           2), ([os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")], None, 1, 3),
                          ([os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")], None, 2, 6)])
def test_filesource_with_watch_false(config, input_files, parser_kwargs, repeat, expected_count):

    pipe = Pipeline(config)

    file_source_stage = FileSource(config, files=input_files, watch=False, parser_kwargs=parser_kwargs, repeat=repeat)
    sink_stage = InMemorySinkStage(config)

    pipe.add_stage(file_source_stage)
    pipe.add_stage(sink_stage)

    pipe.add_edge(file_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count
