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

import pytest

from morpheus.utils.downloader import Downloader


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
@pytest.mark.parametrize('dl_method', ["single_thread", "multiprocess", "multiprocessing", "dask", "dask_thread"])
def test_constructor_download_type(use_env: bool, dl_method: str):

    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_method
    else:
        kwargs['download_method'] = dl_method

    downloader = Downloader(**kwargs)
    assert downloader.download_method == dl_method


@pytest.mark.usefixtures("restore_environ")
def test_constructor_env_wins():
    os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = "multiprocessing"
    downloader = Downloader(download_method="single_thread")
    assert downloader.download_method == "multiprocessing"


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
def test_constructor_invalid_dltype(use_env: bool):
    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = "fake"
    else:
        kwargs['download_method'] = "fake"

    with pytest.raises(ValueError):
        Downloader(**kwargs)


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('dl_method,use_processes', [("dask", True), ("dask_thread", False)])
@mock.patch('dask.config')
@mock.patch('dask.distributed.LocalCluster')
def test_get_dask_cluster(mock_dask_cluster: mock.MagicMock,
                          mock_dask_config: mock.MagicMock,
                          dl_method: str,
                          use_processes: bool):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader = Downloader(download_method=dl_method)
    assert downloader.get_dask_cluster() is mock_dask_cluster

    mock_dask_config.set.assert_called_once()
    mock_dask_cluster.assert_called_once_with(start=True, processes=use_processes)


@mock.patch('dask.config')
@mock.patch('dask.distributed.LocalCluster')
@pytest.mark.parametrize('dl_method', ["dask", "dask_thread"])
def test_close(mock_dask_cluster: mock.MagicMock, mock_dask_config: mock.MagicMock, dl_method: str):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader = Downloader(download_method=dl_method)
    assert downloader.get_dask_cluster() is mock_dask_cluster

    mock_dask_config.set.assert_called_once()

    mock_dask_cluster.close.assert_not_called()
    downloader.close()
    mock_dask_cluster.close.assert_called_once()


@mock.patch('dask.distributed.LocalCluster')
@pytest.mark.parametrize('dl_method', ["single_thread", "multiprocess", "multiprocessing"])
def test_close_dask_cluster_noop(mock_dask_cluster: mock.MagicMock, dl_method: str):
    mock_dask_cluster.return_value = mock_dask_cluster
    downloader = Downloader(download_method=dl_method)

    # Method is a no-op when Dask is not used
    downloader.close()

    mock_dask_cluster.assert_not_called()
    mock_dask_cluster.close.assert_not_called()
