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
    """
    The GNN fraud detection pipeline utilizes the "other" pipeline mode.
    """
    from morpheus.config import PipelineModes
    config.mode = PipelineModes.OTHER
    yield config


@pytest.fixture
def example_dir():
    yield os.path.join(TEST_DIRS.examples_dir, 'gnn_fraud_detection_pipeline')


@pytest.fixture
def training_file(example_dir: str):
    yield os.path.join(example_dir, 'training.csv')


@pytest.fixture
def hinsage_model(example_dir: str):
    yield os.path.join(example_dir, 'model/hinsage-model.pt')


@pytest.fixture
def xgb_model(example_dir: str):
    yield os.path.join(example_dir, 'model/xgb-model.pt')


# Some of the code inside gnn_fraud_detection_pipeline performs some relative imports in the form of:
#    from .mod import Class
# For this reason we need to ensure that the examples dir is in the sys.path first
@pytest.fixture
def gnn_fraud_detection_pipeline(request: pytest.FixtureRequest, restore_sys_path, reset_plugins):
    sys.path.append(TEST_DIRS.examples_dir)
    import gnn_fraud_detection_pipeline
    yield gnn_fraud_detection_pipeline


@pytest.fixture
def test_data():
    """
    Construct test data, a small DF of 10 rows which we will build a graph from
    The nodes in our graph will be the unique values from each of our three columns, and the index is also
    representing our transaction ids.
    There is only one duplicated value (2697) in our dataset so we should expect 29 nodes
    Our expected edges will be each value in client_node and merchant_node to their associated index value ex:
    (795, 2) & (8567, 2)
    thus we should expect 20 edges, although 2697 is duplicated in the client_node column we should expect two
    unique edges for each entry (2697, 14) & (2697, 91)
    """
    import pandas as pd
    index = [2, 14, 16, 26, 41, 42, 70, 91, 93, 95]
    client_data = [795, 2697, 5531, 415, 2580, 3551, 6547, 2697, 3503, 7173]
    merchant_data = [8567, 4609, 2781, 7844, 629, 6915, 7071, 570, 2446, 8110]

    df_data = {
        'index': index,
        'client_node': client_data,
        'merchant_node': merchant_data,
        'fraud_label': [1 for _ in range(len(index))]
    }

    # Fill in the other columns so that we match the shape the model is expecting
    for i in range(1000, 1113):
        # these two values are skipped, apparently place-holders for client_node & merchant_node
        if i not in (1002, 1003):
            df_data[str(i)] = [0 for _ in range(len(index))]

    df = pd.DataFrame(df_data, index=index)

    expected_nodes = set(index + client_data + merchant_data)
    assert len(expected_nodes) == 29  # ensuring test data & assumptions are correct

    expected_edges = set()
    for data in (client_data, merchant_data):
        for (i, val) in enumerate(data):
            expected_edges.add((val, index[i]))

    assert len(expected_edges) == 20  # ensuring test data & assumptions are correct

    yield dict(index=index,
               client_data=client_data,
               merchant_data=merchant_data,
               df=df,
               expected_nodes=expected_nodes,
               expected_edges=expected_edges)
