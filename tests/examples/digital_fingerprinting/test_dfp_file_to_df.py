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
import os
import re
from datetime import datetime
from datetime import timezone

import fsspec
import pytest

from morpheus.common import FileTypes
from morpheus.config import Config
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.column_info import DataFrameInputSchema
from utils import TEST_DIRS


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
