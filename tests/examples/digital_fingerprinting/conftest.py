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

import pytest
import yaml

from utils import TEST_DIRS
from utils import import_or_skip

SKIP_REASON = (
    "Tests for the digital_fingerprinting production example requires a number of packages not installed in the "
    "Morpheus development environment. See `/home/dagardner/work/morpheus/examples/ransomware_detection/README.md` "
    "for details on installing these additional dependencies")


@pytest.fixture(autouse=True, scope='session')
def dask_distributed(fail_missing: bool):
    """
    Mark tests requiring dask.distributed
    """
    yield import_or_skip("dask.distributed", reason=SKIP_REASON, fail_missing=fail_missing)


@pytest.fixture
@pytest.mark.use_python
def config(config):
    """
    The digital_fingerprinting production example utilizes the Auto Encoder config, and requires C++ execution disabled.
    """
    from morpheus.config import ConfigAutoEncoder
    config.ae = ConfigAutoEncoder()
    yield config


@pytest.fixture
def example_dir():
    yield os.path.join(TEST_DIRS.examples_dir, 'digital_fingerprinting/production/morpheus')


# Some of the code inside ransomware_detection performs imports in the form of:
#    from common....
# For this reason we need to ensure that the examples/ransomware_detection dir is in the sys.path first
@pytest.fixture(autouse=True)
def dfp_prod_in_sys_path(request: pytest.FixtureRequest, restore_sys_path, reset_plugins, example_dir):
    sys.path.append(example_dir)


@pytest.fixture
def dfp_message_meta(config, dataset_pandas):
    from dfp.messages.multi_dfp_message import DFPMessageMeta

    user_id = 'test_user'
    df = dataset_pandas['filter_probs.csv']
    df[config.ae.timestamp_column_name] = [1683054498 + i for i in range(0, len(df) * 100, 100)]
    df['user_id'] = user_id
    yield DFPMessageMeta(df, user_id)
