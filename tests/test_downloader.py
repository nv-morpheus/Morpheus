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

import pytest

from morpheus.utils.downloader import Downloader


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize('use_env', [True, False])
@pytest.mark.parametrize('dl_type', ["single_thread", "multiprocess", "multiprocessing", "dask", "dask_thread"])
def test_constructor_download_type(use_env: bool, dl_type: str):

    kwargs = {}
    if use_env:
        os.environ['MORPHEUS_FILE_DOWNLOAD_TYPE'] = dl_type
    else:
        kwargs['download_method'] = dl_type

    downloader = Downloader(**kwargs)
    assert downloader.download_method == dl_type


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
