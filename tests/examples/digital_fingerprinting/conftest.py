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
import sys
import typing
from unittest import mock

import pytest

from _utils import TEST_DIRS
from _utils import import_or_skip
from _utils.dataset_manager import DatasetManager
from morpheus.config import Config

SKIP_REASON = (
    "Tests for the digital_fingerprinting production example requires a number of packages not installed in the "
    "Morpheus development environment.")


@pytest.fixture(autouse=True, scope='session')
def dask_distributed(fail_missing: bool):
    """
    Mark tests requiring dask.distributed
    """
    yield import_or_skip("dask.distributed", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(autouse=True, scope='session')
def dask_cuda(fail_missing: bool):
    """
    Mark tests requiring dask.distributed
    """
    yield import_or_skip("dask_cuda", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(autouse=True, scope='session')
def mlflow(fail_missing: bool):
    """
    Mark tests requiring mlflow
    """
    yield import_or_skip("mlflow", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture(name='ae_feature_cols', scope='session')
def ae_feature_cols_fixture():
    with open(os.path.join(TEST_DIRS.data_dir, 'columns_ae_cloudtrail.txt'), encoding='utf-8') as fh:
        yield [x.strip() for x in fh.readlines()]


@pytest.fixture(name="config")
def config_fixture(config_no_cpp: Config, ae_feature_cols: typing.List[str]):
    """
    The digital_fingerprinting production example utilizes the Auto Encoder config, and requires C++ execution disabled.
    """
    from morpheus.config import ConfigAutoEncoder
    config = config_no_cpp
    config.ae = ConfigAutoEncoder()
    config.ae.feature_columns = ae_feature_cols
    yield config


@pytest.fixture(name="example_dir")
def example_dir_fixture():
    yield os.path.join(TEST_DIRS.examples_dir, 'digital_fingerprinting/production/morpheus')


# Some of the code inside the production DFP example performs imports in the form of:
#    from ..utils.model_cache import ModelCache
# For this reason we need to ensure that the digital_fingerprinting/production/morpheus dir is in sys.path
@pytest.fixture(autouse=True)
def dfp_prod_in_sys_path(
        restore_sys_path: list[str],  # pylint: disable=unused-argument
        reset_plugins: None,  # pylint: disable=unused-argument
        example_dir: str):
    sys.path.append(example_dir)


@pytest.fixture(name="dfp_message_meta")
def dfp_message_meta_fixture(config, dataset_pandas: DatasetManager):
    import pandas as pd
    from dfp.messages.multi_dfp_message import DFPMessageMeta

    user_id = 'test_user'
    df = dataset_pandas['filter_probs.csv']
    df[config.ae.timestamp_column_name] = pd.to_datetime([1683054498 + i for i in range(0, len(df) * 30, 30)], unit='s')
    df[config.ae.userid_column_name] = user_id
    yield DFPMessageMeta(df, user_id)


@pytest.fixture
def dfp_multi_message(dfp_message_meta):
    from dfp.messages.multi_dfp_message import MultiDFPMessage
    yield MultiDFPMessage(meta=dfp_message_meta)


@pytest.fixture
def dfp_multi_ae_message(dfp_message_meta):
    from morpheus.messages.multi_ae_message import MultiAEMessage
    yield MultiAEMessage(meta=dfp_message_meta, model=mock.MagicMock())
