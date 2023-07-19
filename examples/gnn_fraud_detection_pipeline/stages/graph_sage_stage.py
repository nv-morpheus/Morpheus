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
import os
import pickle
import typing

import dgl
import mrc
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

from .graph_construction_stage import FraudGraphMultiMessage
from .model import HinSAGE


@dataclasses.dataclass
class GraphSAGEMultiMessage(MultiMessage):
    node_identifiers: typing.List[int]
    inductive_embedding_column_names: typing.List[str]

    def __init__(self,
                 *,
                 meta: MessageMeta,
                 mess_offset: int = 0,
                 mess_count: int = -1,
                 node_identifiers: typing.List[int],
                 inductive_embedding_column_names: typing.List[str]):
        super().__init__(meta=meta, mess_offset=mess_offset, mess_count=mess_count)

        self.node_identifiers = node_identifiers
        self.inductive_embedding_column_names = inductive_embedding_column_names


@register_stage("gnn-fraud-sage", modes=[PipelineModes.OTHER])
class GraphSAGEStage(SinglePortStage):

    def __init__(self,
                 config: Config,
                 model_dir: str,
                 batch_size: int = 100,
                 record_id: str = "index",
                 target_node: str = "transaction"):
        super().__init__(config)

        self._dgl_model, _, self.hyperparam = self._load_model(model_dir)
        self._batch_size = batch_size
        # self._sample_size = list(sample_size)
        self._record_id = record_id
        self._target_node = target_node

    @property
    def name(self) -> str:
        return "gnn-fraud-sage"

    def accepted_types(self) -> typing.Tuple:
        return (FraudGraphMultiMessage, )

    def supports_cpp_node(self):
        return False

    def _load_model(self, model_dir, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """Load trained model, graph structure from given directory

        Args:
            model_dir (str): directory path for trained model obj.
            device (str): device runtime.

        Returns:
            List[HeteroRGCN, DGLHeteroGraph]: model and graph structure.
        """

        with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
            graph = pickle.load(f)
        with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
            hyperparameters = pickle.load(f)
        model = HinSAGE(graph,
                        in_size=hyperparameters['in_size'],
                        hidden_size=hyperparameters['hidden_size'],
                        out_size=hyperparameters['out_size'],
                        n_layers=hyperparameters['n_layers'],
                        embedding_size=hyperparameters['embedding_size'],
                        target=hyperparameters['target_node']).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))

        return model, graph, hyperparameters

    @torch.no_grad()
    def _evaluate(self, model, eval_loader, feature_tensors, target_node, device='cpu'):
        """Takes trained RGCN model and input dataloader & produce logits and embedding.

        Args:
            model (HeteroRGCN): trained HeteroRGCN model object
            eval_loader (NodeDataLoader): evaluation dataloader
            feature_tensors (torch.Tensor) : test feature tensor
            target_node (str): target node encoding.
            device (str, optional): device runtime. Defaults to 'cpu'.

        Returns:
            List: logits, index & output embedding.
        """
        model.eval()
        eval_logits = []
        eval_seeds = []
        embedding = []

        for _, output_nodes, blocks in eval_loader:

            seed = output_nodes[target_node]

            nid = blocks[0].srcnodes[target_node].data[dgl.NID]
            blocks = [b.to(device) for b in blocks]
            input_features = feature_tensors[nid].to(device)
            logits, embedd = model.infer(blocks, input_features)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seed)
            embedding.append(embedd)

        eval_logits = torch.cat(eval_logits)
        eval_seeds = torch.cat(eval_seeds)
        embedding = torch.cat(embedding)
        return eval_logits, eval_seeds, embedding

    def inference(self,
                  model,
                  input_graph,
                  feature_tensors,
                  test_idx,
                  target_node="transaction",
                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """Minibatch inference on test graph

        Args:
            model (HeteroRGCN) : trained HeteroRGCN model.
            input_graph (DGLHeterograph) : test graph
            feature_tensors (torch.Tensor) : node features
            test_idx (list): test index
            target_node (list): target node
            device (str, optional): device runtime.

        Returns:
            list: logits, index, output embedding
        """

        # create sampler and test dataloaders
        full_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=[4, 3])
        test_dataloader = dgl.dataloading.DataLoader(input_graph, {target_node: test_idx},
                                                     full_sampler,
                                                     batch_size=self._batch_size,
                                                     shuffle=False,
                                                     drop_last=False,
                                                     num_workers=0)
        _, _, test_embedding = self._evaluate(model, test_dataloader, feature_tensors, target_node, device=device)

        return test_embedding

    def _process_message(self, message: FraudGraphMultiMessage):

        node_identifiers = list(message.get_meta(self._record_id).to_pandas())

        # inductive_embedding = self._inductive_step_hinsage(message.graph, self._keras_model, node_identifiers)
        inductive_embedding = self.inference(self._dgl_model, message.graph, message.node_features, node_identifiers)

        inductive_embedding = cudf.DataFrame(inductive_embedding)
        # Rename the columns to be more descriptive
        inductive_embedding.rename(lambda x: "ind_emb_" + str(x), axis=1, inplace=True)

        for col in inductive_embedding.columns.values.tolist():
            message.set_meta(col, inductive_embedding[col])

        assert (message.mess_count == len(inductive_embedding))

        return GraphSAGEMultiMessage(meta=message.meta,
                                     node_identifiers=node_identifiers,
                                     inductive_embedding_column_names=inductive_embedding.columns.values.tolist(),
                                     mess_offset=message.mess_offset,
                                     mess_count=message.mess_count)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, ops.map(self._process_message))
        builder.make_edge(input_stream[0], node)
        return node, GraphSAGEMultiMessage
