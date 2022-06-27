# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import dataclasses
import typing

import networkx as nx
import pandas as pd
import srf
from stellargraph import StellarGraph

import cudf

from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@dataclasses.dataclass
class FraudGraphMultiMessage(MultiMessage):
    graph: StellarGraph


class FraudGraphConstructionStage(SinglePortStage):

    def __init__(self, c: Config, training_file: str):
        super().__init__(c)
        self._training_data = cudf.read_csv(training_file)
        self._column_names = self._training_data.columns.values.tolist()

    @property
    def name(self) -> str:
        return "fraud-graph-construction"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def supports_cpp_node():
        return False

    @staticmethod
    def _graph_construction(nodes, edges, node_features) -> StellarGraph:
        g_nx = nx.Graph()

        # add nodes
        for key, values in nodes.items():
            g_nx.add_nodes_from(values, ntype=key)
        # add edges
        for edge in edges:
            g_nx.add_edges_from(edge)

        return StellarGraph(g_nx, node_type_name="ntype", node_features=node_features)

    @staticmethod
    def _build_graph_features(dataset: pd.DataFrame) -> StellarGraph:

        nodes = {
            "client": dataset.client_node,
            "merchant": dataset.merchant_node,
            "transaction": dataset.index,
        }

        edges = [
            zip(dataset.client_node, dataset.index),
            zip(dataset.merchant_node, dataset.index),
        ]

        transaction_node_data = dataset.drop(["client_node", "merchant_node", "fraud_label", "index"], axis=1)
        client_node_data = pd.DataFrame([1] * len(dataset.client_node.unique())).set_index(dataset.client_node.unique())
        merchant_node_data = pd.DataFrame([1] * len(dataset.merchant_node.unique())).set_index(
            dataset.merchant_node.unique())

        node_features = {
            "transaction": transaction_node_data,
            "client": client_node_data,
            "merchant": merchant_node_data,
        }

        return FraudGraphConstructionStage._graph_construction(nodes, edges, node_features)

    def _process_message(self, message: MultiMessage):
        graph_data = cudf.concat([self._training_data, message.get_meta(self._column_names)])
        graph_data = graph_data.set_index(graph_data['index'])
        graph = FraudGraphConstructionStage._build_graph_features(graph_data.to_pandas())
        return FraudGraphMultiMessage(meta=message.meta,
                                      graph=graph,
                                      mess_offset=message.mess_offset,
                                      mess_count=message.mess_count)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self._process_message)
        builder.make_edge(input_stream[0], node)
        return node, FraudGraphMultiMessage
