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

import functools
import json
import os
import re
import types
from datetime import datetime
from datetime import timezone
from unittest import mock

import fsspec
import pytest

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from utils import TEST_DIRS


def test_single_object_to_dataframe():
    from dfp.stages.dfp_file_to_df import _single_object_to_dataframe

    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              'appshield',
                              'snapshot-1',
                              'threadlist_2022-01-30_10-26-01.670391.json')
    file_specs = fsspec.open_files(input_file)
    assert len(file_specs)

    file_obj = fsspec.core.OpenFile(fs=file_specs.fs, path=file_specs[0].path)

    schema = DataFrameInputSchema(
        column_info=[ColumnInfo(name='titles', dtype=str), ColumnInfo(name='data', dtype=str)])
    df = _single_object_to_dataframe(file_obj, schema, FileTypes.Auto, False, {})

    assert sorted(df.columns) == ['data', 'titles']
    assert df['titles'].to_list() == [["TID", "Offset", "State", "WaitReason", "PID", "Process"]]

    with open(input_file, encoding='UTF-8') as fh:
        d = json.load(fh)
        expected_data = d['data']

    df['data'].to_list() == expected_data


def test_single_object_to_dataframe_timeout():
    from dfp.stages.dfp_file_to_df import _single_object_to_dataframe

    input_glob = os.path.join(TEST_DIRS.tests_data_dir, 'appshield', 'snapshot-1', 'fake_wont_match*.json')
    bad_file = fsspec.core.OpenFile(fs=fsspec.open_files(input_glob).fs, path='/tmp/fake/doesnt/exit.csv')

    assert _single_object_to_dataframe(bad_file, DataFrameInputSchema(), FileTypes.CSV, False, {}) is None


@pytest.mark.restore_environ
def test_constructor(config: Config):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage

    # The user may have this already set, ensure it is undefined
    os.environ.pop('MORPHEUS_FILE_DOWNLOAD_TYPE', None)

    schema = DataFrameInputSchema()
    stage = DFPFileToDataFrameStage(config,
                                    schema,
                                    filter_null=False,
                                    file_type=FileTypes.PARQUET,
                                    parser_kwargs={'test': 'this'},
                                    cache_dir='/test/path/cache')

    assert isinstance(stage, SinglePortStage)
    assert isinstance(stage, PreallocatorMixin)
    assert stage._schema is schema
    assert stage._file_type == FileTypes.PARQUET
    assert not stage._filter_null
    assert stage._parser_kwargs == {'test': 'this'}
    assert stage._cache_dir.startswith('/test/path/cache')
    assert stage._dask_cluster is None
    assert stage._download_method == "dask_thread"


@pytest.mark.restore_environ
@pytest.mark.parametrize('dl_type', ["single_thread", "multiprocess", "dask", "dask_thread"])
def test_constructor_download_type(config: Config, dl_type: str):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage

    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_type
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema())
    assert stage._download_method == dl_type


@pytest.mark.restore_environ
@pytest.mark.parametrize('dl_type,use_processes', [("dask", True), ("dask_thread", False)])
@mock.patch('dask.config')
@mock.patch('dfp.stages.dfp_file_to_df.LocalCluster')
def test_get_dask_cluster(mock_dask_cluster: mock.MagicMock,
                          mock_dask_config: mock.MagicMock,
                          config: Config,
                          dl_type: str,
                          use_processes: bool):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
    mock_dask_cluster.return_value = mock_dask_cluster

    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_type
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema())
    assert stage._get_dask_cluster() is mock_dask_cluster

    mock_dask_config.set.assert_called_once()
    mock_dask_cluster.assert_called_once_with(start=True, processes=use_processes)


@mock.patch('dask.config')
@mock.patch('dfp.stages.dfp_file_to_df.LocalCluster')
def test_close_dask_cluster(mock_dask_cluster: mock.MagicMock, mock_dask_config: mock.MagicMock, config: Config):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
    mock_dask_cluster.return_value = mock_dask_cluster
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema())
    assert stage._get_dask_cluster() is mock_dask_cluster

    mock_dask_config.set.assert_called_once()

    mock_dask_cluster.close.assert_not_called()
    stage._close_dask_cluster()
    mock_dask_cluster.close.assert_called_once()


@mock.patch('dfp.stages.dfp_file_to_df.LocalCluster')
def test_close_dask_cluster_noop(mock_dask_cluster: mock.MagicMock, config: Config):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
    mock_dask_cluster.return_value = mock_dask_cluster
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema())

    # Method is a no-op when Dask is not used
    assert stage._dask_cluster is None
    stage._close_dask_cluster()

    mock_dask_cluster.assert_not_called()
    mock_dask_cluster.close.assert_not_called()
