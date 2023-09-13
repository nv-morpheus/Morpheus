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
from unittest import mock

import fsspec
import pytest

import cudf

from _utils import TEST_DIRS
from morpheus.common import FileTypes
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.input.file_source import FileSource
from morpheus.stages.output.in_memory_sink_stage import InMemorySinkStage


@pytest.mark.use_python
def test_constructor(config):
    file_source = FileSource(
        config,
        files=["path/to/*.json"],
        watch=True,
        sort=True,
        file_type=FileTypes.JSON,
        parser_kwargs={"key": "value"},
        watch_interval=2.0,
    )

    assert file_source._files == ["path/to/*.json"]
    assert file_source._watch
    assert file_source._sort
    assert file_source._file_type == FileTypes.JSON
    assert file_source._parser_kwargs == {"key": "value"}
    assert file_source._watch_interval == 2.0


@pytest.mark.use_python
@pytest.mark.parametrize("input_files", [["file1.json", "file2.json"], []])
def test_constructor_with_invalid_params(config, input_files):
    with pytest.raises(ValueError):
        # 'watch' is True, but multiple files are provided
        FileSource(config, files=input_files, watch=True)


@pytest.mark.use_python
@pytest.mark.parametrize("input_file,filetypes,parser_kwargs,expected_df_count",
                         [("filter_probs.json", FileTypes.Auto, {
                             "lines": False
                         }, 20), ("filter_probs.jsonlines", FileTypes.JSON, {
                             "lines": True
                         }, 20)])
def test_generate_frames(input_file, filetypes, parser_kwargs, expected_df_count):
    in_file = fsspec.open(os.path.join(TEST_DIRS.tests_data_dir, input_file))

    meta = FileSource.generate_frames(file=in_file, file_type=filetypes, parser_kwargs=parser_kwargs)

    assert len(meta.df.columns) == 4
    assert len(meta.df) == expected_df_count
    assert isinstance(meta, MessageMeta)
    assert isinstance(meta.df, cudf.DataFrame)


@pytest.mark.use_python
@pytest.mark.parametrize("input_files,parser_kwargs,expected_count",
                         [([
                             "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T00_05_06.806Z.json",
                             "s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022-08-01T12_09_47.901Z.json"
                         ], {
                             "lines": False, "orient": "records"
                         },
                           2), ([os.path.join(TEST_DIRS.tests_data_dir, "triton_*.csv")], None, 3)])
def test_filesource_with_watch_false(config, input_files, parser_kwargs, expected_count):

    pipe = Pipeline(config)

    file_source_stage = FileSource(config, files=input_files, watch=False, parser_kwargs=parser_kwargs)
    sink_stage = InMemorySinkStage(config)

    pipe.add_stage(file_source_stage)
    pipe.add_stage(sink_stage)

    pipe.add_edge(file_source_stage, sink_stage)

    pipe.run()

    assert len(sink_stage.get_messages()) == expected_count


@pytest.mark.use_python
def test_build_source_watch_remote_files(config):
    files = ["s3://rapidsai-data/cyber/morpheus/dfp/duo/DUO_2022*.json"]
    source = FileSource(config=config, files=files, watch=True)

    mock_node = mock.MagicMock()
    mock_builder = mock.MagicMock()
    mock_builder.make_source.return_value = mock_node
    out_stream, out_type = source._build_source(mock_builder)

    assert out_stream == mock_node
    assert out_type == fsspec.core.OpenFiles
