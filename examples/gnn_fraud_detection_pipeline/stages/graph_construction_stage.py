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

import dataclasses
import pathlib
import typing

import dgl
import mrc
import pandas as pd
import torch
from mrc.core import operators as ops

import cudf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@dataclasses.dataclass
class FraudGraphMultiMessage(MultiMessage):
    graph: "dgl.heterograph.DGLGraph"
    node_features: "torch.tensor"

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 graph: "dgl.heterograph.DGLGraph",
                 node_features="torch.tensor",
                 test_index: "torch.tensor"):
        super().__init__(meta=meta, mess_offset=mess_offset, mess_count=mess_count)

        self.graph = graph
        self.node_features = node_features
        self.test_index = test_index


@register_stage("fraud-graph-construction", modes=[PipelineModes.OTHER])
class FraudGraphConstructionStage(SinglePortStage):

    def __init__(self, config: Config, training_file: pathlib.Path, input_file: pathlib.Path):
        """
        Create a fraud-graph-construction stage

        Parameters
        ----------
        c : Config
            The Morpheus config object
        training_file : pathlib.Path, exists = True, dir_okay = False
            A CSV training file to load to seed the graph
        """
        super().__init__(config)
        self._training_data = cudf.read_csv(training_file)
        self._training_data = training_file
        self._input_file = input_file
        # self._column_names = self._training_data.columns.values.tolist()

    @property
    def name(self) -> str:
        return "fraud-graph-construction"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, )

    def supports_cpp_node(self):
        return False

    @staticmethod
    def _prepare_data(training_data, test_data):
        """Process data for training/inference operation

        Parameters
        ----------
        training_data : str
            path to training data
        test_data : str
            path to test/validation data

        Returns
        -------
        tuple
        tuple of (test_index, whole data)
        """

        df_train = pd.read_csv(training_data)
        train_idx_ = df_train.shape[0]
        df_test = pd.read_csv(test_data)
        df = pd.concat([df_train, df_test], axis=0)
        df['tran_id'] = df['index']

        def map_node_id(df, col_name):
            """ Convert column node list to integer index for dgl graph.

            Args:
                df (pd.DataFrame): dataframe
                col_name (list) : column list
            """
            node_index = {j: i for i, j in enumerate(df[col_name].unique())}
            df[col_name] = df[col_name].map(node_index)

        meta_cols = ['tran_id', 'client_node', 'merchant_node']
        for col in meta_cols:
            map_node_id(df, col)

        test_idx = df['tran_id'][train_idx_:]

        df['index'] = df['tran_id']
        df.index = df['index']
        return test_idx, df

    @staticmethod
    def _build_graph_features(dataset: pd.DataFrame) -> "dgl.heterograph.DGLGraph":

        edge_list = {
            ('client', 'buy', 'transaction'): (dataset['client_node'].values, dataset['index'].values),
            ('transaction', 'bought', 'client'): (dataset['index'].values, dataset['client_node'].values),
            ('transaction', 'issued', 'merchant'): (dataset['index'].values, dataset['merchant_node'].values),
            ('merchant', 'sell', 'transaction'): (dataset['merchant_node'].values, dataset['index'].values)
        }
        col_drop = ['client_node', 'merchant_node', 'index', 'fraud_label', 'tran_id']
        node_features = torch.from_numpy(dataset.drop(col_drop, axis=1).values).float()
        node_features = (node_features - node_features.mean(0)) / (0.0001 + node_features.std(0))
        graph = dgl.heterograph(edge_list)

        return graph, node_features

    def _process_message(self, message: MultiMessage):
        test_index, graph_data = self._prepare_data(self._training_data, self._input_file)
        graph, node_features = FraudGraphConstructionStage._build_graph_features(graph_data)
        return FraudGraphMultiMessage.from_message(message,
                                                   graph=graph,
                                                   node_features=node_features,
                                                   test_index=test_index)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)
        return node, FraudGraphMultiMessage
