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

import torch
from dgl import function as fn
from torch import nn
from torch.nn import functional as F


class HeteroRGCNLayer(nn.Module):
    """Relational graph convolutional layer

    Args:
        in_size (int): input feature size.
        out_size (int): output feature size.
        etypes (list): edge relation names.
    """

    def __init__(self, in_size, out_size, etypes):
        super().__init__()
        # W_r for each relation
        input_sizes = [in_size] * len(etypes) if isinstance(in_size, int) else in_size
        self.weight = nn.ModuleDict({name: nn.Linear(in_dim, out_size) for name, in_dim in zip(etypes, input_sizes)})

    def forward(self, graph, feat_dict):
        """Forward computation

        Args:
            graph (DGLHeterograph): Input graph
            feat_dict (dict[str, torch.Tensor]): Node features for each node.

        Returns:
            dict[str, torch.Tensor]: New node features for each node type.
        """

        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, _ in graph.canonical_etypes:
            # Compute W_r * h
            if srctype in feat_dict:
                w_h = self.weight[etype](feat_dict[srctype])
                # Save it in graph for message passing
                graph.nodes[srctype].data[f'Wh_{etype}'] = w_h
                funcs[etype] = (fn.copy_u(f'Wh_{etype}', 'm'), fn.mean('m', 'h'))

        graph.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: graph.dstnodes[ntype].data['h'] for ntype in graph.ntypes if 'h' in graph.dstnodes[ntype].data}


class HeteroRGCN(nn.Module):
    """Relational graph convolutional layer

    Args:
        graph (DGLHeterograph): input graph.
        in_size (int): input feature size.
        hidden_size (int): hidden layer size.
        out_size (int): output feature size.
        n_layers (int): number of layers.
        embedding_size (int): embedding size
        device (str, optional): host device. Defaults to 'cpu'.
        target (str, optional): target node. Defaults to 'transaction'.
    """

    def __init__(self,
                 graph,
                 in_size,
                 hidden_size,
                 out_size,
                 n_layers,
                 embedding_size,
                 device='cpu',
                 target='transaction'):

        super().__init__()

        self.target = target
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(graph.number_of_nodes(ntype), embedding_size))
            for ntype in graph.ntypes if ntype != self.target
        }
        for _, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed_dict = {ntype: embedding.to(device) for ntype, embedding in embed_dict.items()}
        self.device = device
        self.g_embed = None

        # create layers
        in_sizes = [in_size if src_type == self.target else embedding_size for src_type, _, _ in graph.canonical_etypes]
        layers = [HeteroRGCNLayer(in_sizes, hidden_size, graph.etypes)]

        # hidden layers
        for _ in range(n_layers - 1):
            layers.append(HeteroRGCNLayer(hidden_size, hidden_size, graph.etypes))

        # output layer
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.Sequential(*layers)

    def embed(self, graph, features):
        """Embeddings for all node types.

        Args:
            graph (DGLHeterograph): Input graph
            features (torch.Tensor): target node features

        Returns:
            list: target node embedding
        """
        # get embeddings for all node types. Initialize nodes with random weight.
        h_dict = {self.target: features}
        for ntype in self.embed_dict:
            if graph[0].number_of_nodes(ntype) > 0:
                h_dict[ntype] = self.embed_dict[ntype][graph[0].nodes(ntype).to(self.device)]

        # Forward pass to layers.
        for i, layer in enumerate(self.layers[:-1]):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(graph[i], h_dict)
        self.g_embed = h_dict
        return h_dict[self.target]

    def forward(self, graph, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            graph (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature
        Returns:
            list: layer embedding
        """
        return self.layers[-1](self.embed(graph, features))

    def infer(self, graph, features):
        """Perform forward inference on graph G with feature tensor input

        Args:
            graph (DGLHeterograph): DGL test graph
            features (torch.Tensor): input feature

        Returns:
            list: logits, embedding vector
        """
        embedding = self.embed(graph, features)
        predictions = self.layers[-1](embedding)
        return nn.Sigmoid()(predictions), embedding
