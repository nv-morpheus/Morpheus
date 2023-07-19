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

from dgl import nn as dglnn
from torch import nn
from torch.nn import functional as F


class HeteroRGCN(nn.Module):
    """HeteroRGCN

        Args:
        g (_type_): _description_
        in_size (_type_): _description_
        hidden_size (_type_): _description_
        out_size (_type_): _description_
        n_layers (_type_): _description_
        embedding_size (_type_): _description_
        device (str, optional): _description_. Defaults to 'cpu'.
        target (str, optional): _description_. Defaults to 'transaction'.
    """

    def __init__(self, input_graph, in_size, hidden_size, out_size, n_layers, embedding_size, target='transaction'):

        super().__init__()

        self.target = target

        # categorical embeding
        self.hetro_embedding = dglnn.HeteroEmbedding(
            {ntype: input_graph.number_of_nodes(ntype)
             for ntype in input_graph.ntypes if ntype != self.target},
            embedding_size)

        # input size
        in_sizes = {
            rel: in_size if src_type == target else embedding_size
            for src_type, rel, _ in input_graph.canonical_etypes
        }

        self.layers = nn.ModuleList()

        self.layers.append(
            dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_sizes[rel], hidden_size)
                                   for rel in input_graph.etypes},
                                  aggregate='sum'))

        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hidden_size, hidden_size)
                                       for rel in input_graph.etypes},
                                      aggregate='sum'))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, input_graph, features):
        """Perform forward inference on graph g with feature tensor input

        Args:
            input_graph (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature
        Returns:
            list: layer embedding
        """

        # get embeddings for all node types. Initialize nodes with random weights.
        h_dict = self.hetro_embedding(
            {ntype: input_graph[0].nodes(ntype)
             for ntype in self.hetro_embedding.embeds.keys()})

        h_dict[self.target] = features

        # Forward pass to layers.
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(input_graph[i], h_dict)

        embedding = h_dict[self.target]

        return self.layers[-1](embedding), embedding

    def infer(self, input_graph, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            input_graph (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature

        Returns:
            list: logits, embedding vector
        """
        predictions, embedding = self(input_graph, features)
        return nn.Sigmoid()(predictions), embedding


class HinSAGE(nn.Module):

    def __init__(self,
                 g,
                 in_size,
                 hidden_size,
                 out_size,
                 n_layers,
                 embedding_size,
                 target='transaction',
                 aggregator_type='mean'):

        super().__init__()

        self.target = target

        # categorical embeding
        self.hetro_embedding = dglnn.HeteroEmbedding(
            {ntype: g.number_of_nodes(ntype)
             for ntype in g.ntypes if ntype != self.target}, embedding_size)

        self.layers = nn.ModuleList()

        # create input features
        in_feats = {rel: embedding_size if rel != self.target else in_size for rel in g.ntypes}

        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    rel: dglnn.SAGEConv(
                        (in_feats[src_type], in_feats[v_type]), hidden_size, aggregator_type=aggregator_type)
                    for src_type,
                    rel,
                    v_type in g.canonical_etypes
                },
                aggregate='sum'))

        for _ in range(n_layers - 1):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.SAGEConv(
                            hidden_size,
                            hidden_size,
                            aggregator_type=aggregator_type,
                        )
                        for rel in g.etypes
                    },
                    aggregate='sum'))

        # output layer
        self.layers.append(nn.Linear(hidden_size, out_size))

    def forward(self, input_graph, features):
        """Embeddings for all node types.

        Args:
            input_graph (DGLHeterograph): Input graph
            features (torch.Tensor): target node features

        Returns:
            list: target node embedding
        """

        h_dict = self.hetro_embedding(
            {ntype: input_graph[0].nodes(ntype)
             for ntype in self.hetro_embedding.embeds.keys()})

        h_dict[self.target] = features

        # Forward pass to layers.
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(input_graph[i], h_dict)

        embedding = h_dict[self.target]
        out = self.layers[-1](embedding)
        return out, embedding

    def infer(self, input_graph, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            input_graph (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature

        Returns:
            list: logits, embedding vector
        """
        predictions, embedding = self(input_graph, features)
        return nn.Sigmoid()(predictions), embedding
