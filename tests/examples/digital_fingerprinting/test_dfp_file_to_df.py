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

import hashlib
import json
import os
from functools import partial
from unittest import mock

import fsspec
import pandas as pd
import pytest

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from utils import TEST_DIRS
from utils.dataset_manager import DatasetManager


@pytest.fixture
def single_file_obj():
    input_file = os.path.join(TEST_DIRS.tests_data_dir,
                              'appshield',
                              'snapshot-1',
                              'threadlist_2022-01-30_10-26-01.670391.json')
    file_specs = fsspec.open_files(input_file)
    assert len(file_specs) == 1
    yield fsspec.core.OpenFile(fs=file_specs.fs, path=file_specs[0].path)


def test_single_object_to_dataframe(single_file_obj: fsspec.core.OpenFile):
    from dfp.stages.dfp_file_to_df import _single_object_to_dataframe

    schema = DataFrameInputSchema(
        column_info=[CustomColumn(name='data', dtype=str, process_column_fn=lambda df: df['data'].to_list()[0])])
    df = _single_object_to_dataframe(single_file_obj, schema, FileTypes.Auto, False, {})

    assert df.columns == ['data']
    with open(single_file_obj.path, encoding='UTF-8') as fh:
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


@pytest.mark.restore_environ
@pytest.mark.parametrize('dl_type', ["single_thread", "multiprocess", "dask", "dask_thread"])
@mock.patch('multiprocessing.get_context')
@mock.patch('dask.config')
@mock.patch('dfp.stages.dfp_file_to_df.Client')
@mock.patch('dfp.stages.dfp_file_to_df.LocalCluster')
@mock.patch('dfp.stages.dfp_file_to_df._single_object_to_dataframe')
def test_get_or_create_dataframe_from_s3_batch_cache_miss(mock_obf_to_df: mock.MagicMock,
                                                          mock_dask_cluster: mock.MagicMock,
                                                          mock_dask_client: mock.MagicMock,
                                                          mock_dask_config: mock.MagicMock,
                                                          mock_mp_gc: mock.MagicMock,
                                                          config: Config,
                                                          dl_type: str,
                                                          tmp_path: str,
                                                          single_file_obj: fsspec.core.OpenFile,
                                                          dataset_pandas: DatasetManager):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
    config.ae.timestamp_column_name = 'v1'
    mock_dask_cluster.return_value = mock_dask_cluster
    mock_dask_client.return_value = mock_dask_client
    mock_dask_client.__enter__.return_value = mock_dask_client
    mock_dask_client.__exit__.return_value = False

    mock_mp_gc.return_value = mock_mp_gc
    mock_mp_pool = mock.MagicMock()
    mock_mp_gc.Pool.return_value = mock_mp_pool
    mock_mp_pool.return_value = mock_mp_pool
    mock_mp_pool.__enter__.return_value = mock_mp_pool
    mock_mp_pool.__exit__.return_value = False

    expected_hash = hashlib.md5(json.dumps([{
        'ukey': single_file_obj.fs.ukey(single_file_obj.path)
    }]).encode()).hexdigest()

    expected_df = dataset_pandas['filter_probs.csv']
    expected_df.sort_values(by=['v1'], inplace=True)
    expected_df.reset_index(drop=True, inplace=True)
    expected_df['batch_count'] = 1
    expected_df["origin_hash"] = expected_hash

    # We're going to feed the function a file object pointing to a different file than the one we are going to return
    # from out mocked fetch function. This way we will be able to easily tell if our mocks are working. Mostly we just
    # want to make sure that we aren't accidentally spinning up dask clusters or process pools in CI
    returnd_df = dataset_pandas['filter_probs.csv']
    if dl_type.startswith('dask'):
        mock_dask_client.gather.return_value = [returnd_df]
    elif dl_type == "multiprocess":
        mock_mp_pool.map.return_value = [returnd_df]
    else:
        mock_obf_to_df.return_value = returnd_df

    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_type
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema(), cache_dir=tmp_path)

    batch = fsspec.core.OpenFiles([single_file_obj], fs=single_file_obj.fs)
    (output_df, cache_hit) = stage._get_or_create_dataframe_from_s3_batch((batch, 1))

    if dl_type == "multiprocess":
        mock_mp_gc.assert_called_once()
        mock_mp_pool.map.assert_called_once()
    else:
        mock_mp_gc.assert_not_called()
        mock_mp_pool.map.assert_not_called()

    if dl_type == "single_thread":
        mock_obf_to_df.assert_called_once()
    else:
        mock_obf_to_df.assert_not_called()

    if dl_type.startswith('dask'):
        mock_dask_client.assert_called_once_with(mock_dask_cluster)
        mock_dask_client.map.assert_called_once()
        mock_dask_client.gather.assert_called_once()
    else:
        mock_dask_cluster.assert_not_called()
        mock_dask_client.assert_not_called()
        mock_dask_config.assert_not_called()

    assert not cache_hit
    dataset_pandas.assert_df_equal(output_df, expected_df)

    expected_cache_file_path = os.path.join(stage._cache_dir, "batches", f"{expected_hash}.pkl")
    assert os.path.exists(expected_cache_file_path)
    dataset_pandas.assert_df_equal(pd.read_pickle(expected_cache_file_path),
                                   expected_df[dataset_pandas['filter_probs.csv'].columns])


@pytest.mark.restore_environ
@pytest.mark.parametrize('dl_type', ["single_thread", "multiprocess", "dask", "dask_thread"])
@mock.patch('multiprocessing.get_context')
@mock.patch('dask.config')
@mock.patch('dfp.stages.dfp_file_to_df.Client')
@mock.patch('dfp.stages.dfp_file_to_df.LocalCluster')
@mock.patch('dfp.stages.dfp_file_to_df._single_object_to_dataframe')
def test_get_or_create_dataframe_from_s3_batch_none_noop(mock_obf_to_df: mock.MagicMock,
                                                         mock_dask_cluster: mock.MagicMock,
                                                         mock_dask_client: mock.MagicMock,
                                                         mock_dask_config: mock.MagicMock,
                                                         mock_mp_gc: mock.MagicMock,
                                                         config: Config,
                                                         dl_type: str,
                                                         tmp_path: str):
    from dfp.stages.dfp_file_to_df import DFPFileToDataFrameStage
    mock_dask_cluster.return_value = mock_dask_cluster
    mock_dask_client.return_value = mock_dask_client

    mock_mp_gc.return_value = mock_mp_gc
    mock_mp_pool = mock.MagicMock()
    mock_mp_gc.Pool.return_value = mock_mp_pool

    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_type
    stage = DFPFileToDataFrameStage(config, DataFrameInputSchema(), cache_dir=tmp_path)
    assert stage._get_or_create_dataframe_from_s3_batch(None) == (None, False)

    mock_obf_to_df.assert_not_called()
    mock_dask_cluster.assert_not_called()
    mock_dask_client.assert_not_called()
    mock_dask_config.assert_not_called()
    mock_mp_gc.assert_not_called()
    mock_mp_pool.map.assert_not_called()

    assert os.listdir(tmp_path) == []
