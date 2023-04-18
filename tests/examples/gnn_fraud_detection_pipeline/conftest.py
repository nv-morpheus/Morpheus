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

import pytest

from utils import TEST_DIRS

SKIP_REASON = ("Tests for the gnn_fraud_detection_pipeline example require a number of packages not installed in the "
               "Morpheus development environment. See `examples/gnn_fraud_detection_pipeline/README.md` for details on "
               "installing these additional dependencies")


@pytest.fixture(autouse=True, scope='session')
def stellargraph():
    """
    All of the fixtures in this subdir require stellargraph
    """
    yield pytest.importorskip("stellargraph", reason=SKIP_REASON)


@pytest.fixture(autouse=True, scope='session')
def cuml():
    """
    All of the fixtures in this subdir require cuml
    """
    yield pytest.importorskip("cuml", reason=SKIP_REASON)


@pytest.fixture(autouse=True, scope='session')
def tensorflow():
    """
    All of the fixtures in this subdir require tensorflow
    """
    yield pytest.importorskip("tensorflow", reason=SKIP_REASON)


@pytest.fixture
def config(config):
    from morpheus.config import PipelineModes
    config.mode = PipelineModes.OTHER
    yield config


@pytest.fixture
def example_dir():
    yield os.path.join(TEST_DIRS.examples_dir, 'gnn_fraud_detection_pipeline')


@pytest.fixture
def training_file(example_dir):
    yield os.path.join(example_dir, 'training.csv')
