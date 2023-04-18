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
import typing
from io import StringIO

import pandas as pd
import pytest

import cudf

from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from utils import TEST_DIRS


@pytest.mark.use_python
@pytest.mark.import_mod(
    [os.path.join(TEST_DIRS.examples_dir, 'gnn_fraud_detection_pipeline/stages/graph_construction_stage.py')])
class TestGraphConstructionStage:

    def test_constructor(config: Config, training_file: str, import_mod: typing.List[typing.Any]):
        graph_construction_stage = import_mod[0]
        stage = graph_construction_stage.FraudGraphConstructionStage(config, training_file)
        assert isinstance(stage._training_data, cudf.DataFrame)

        # The training datafile contains many more columns than this, but these are the four columns
        # that are depended upon in the code
        assert {'client_node', 'index', 'fraud_label', 'merchant_node'}.issubset(stage._column_names)

    def _build_test_data(self):
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
        index = [2, 14, 16, 26, 41, 42, 70, 91, 93, 95]
        client_data = [795, 2697, 5531, 415, 2580, 3551, 6547, 2697, 3503, 7173]
        merchant_data = [8567, 4609, 2781, 7844, 629, 6915, 7071, 570, 2446, 8110]
        df = pd.DataFrame(
            {
                'index': index,
                'client_node': client_data,
                'merchant_node': merchant_data,
                'fraud_label': [1 for _ in range(len(index))]
            },
            index=index)

        expected_nodes = set(index + client_data + merchant_data)
        assert len(expected_nodes) == 29  # ensuring test data & assumptions are correct

        expected_edges = set()
        for data in (client_data, merchant_data):
            for (i, val) in enumerate(data):
                expected_edges.add((val, index[i]))

        assert len(expected_edges) == 20  # ensuring test data & assumptions are correct

        return dict(index=index,
                    client_data=client_data,
                    merchant_data=merchant_data,
                    df=df,
                    expected_nodes=expected_nodes,
                    expected_edges=expected_edges)

    def _check_graph(self, stellargraph: typing.Any, sg: "stellargraph.StellarGraph", expected_nodes, expected_edges):
        assert isinstance(sg, stellargraph.StellarGraph)
        sg.check_graph_for_ml(features=True, expensive_check=True)  # this will raise if it doesn't pass
        assert not sg.is_directed()

        nodes = sg.nodes()
        assert set(nodes) == expected_nodes

        edges = sg.edges()
        assert set(edges) == expected_edges

    def test_graph_construction(self, import_mod: typing.List[typing.Any], stellargraph: typing.Any):
        graph_construction_stage = import_mod[0]

        test_data = self._build_test_data()

        df = test_data['df']
        client_features = pd.DataFrame({0: 1}, index=list(set(test_data['client_data'])))
        merchant_features = pd.DataFrame({0: 1}, index=test_data['merchant_data'])

        # Call _graph_construction
        sg = graph_construction_stage.FraudGraphConstructionStage._graph_construction(
            nodes={
                'client': df.client_node, 'merchant': df.merchant_node, 'transaction': df.index
            },
            edges=[
                zip(df.client_node, df.index),
                zip(df.merchant_node, df.index),
            ],
            node_features={
                "transaction": df[['client_node', 'merchant_node']],
                "client": client_features,
                "merchant": merchant_features
            })

        self._check_graph(stellargraph, sg, test_data['expected_nodes'], test_data['expected_edges'])

    def test_build_graph_features(self, import_mod: typing.List[typing.Any], stellargraph: typing.Any):
        graph_construction_stage = import_mod[0]

        test_data = self._build_test_data()
        sg = graph_construction_stage.FraudGraphConstructionStage._build_graph_features(test_data['df'])
        self._check_graph(stellargraph, sg, test_data['expected_nodes'], test_data['expected_edges'])

    def test_process_message(self, config: Config, import_mod: typing.List[typing.Any], stellargraph: typing.Any):
        graph_construction_stage = import_mod[0]
        test_data = self._build_test_data()

        df = test_data['df']

        # The stage wants a csv file from the first 5 rows
        training_data = StringIO(df[0:5].to_csv(index=False))
        stage = graph_construction_stage.FraudGraphConstructionStage(config, training_data)

        # Since we used the first 5 rows as the training data, send the second 5 as inference data
        meta = MessageMeta(cudf.DataFrame(df))
        mm = MultiMessage(meta=meta, mess_offset=5, mess_count=5)
        fgmm = stage._process_message(mm)

        assert isinstance(fgmm, graph_construction_stage.FraudGraphMultiMessage)
        assert fgmm.meta is meta
        assert fgmm.mess_offset == 5
        assert fgmm.mess_count == 5

        self._check_graph(stellargraph, fgmm.graph, test_data['expected_nodes'], test_data['expected_edges'])
