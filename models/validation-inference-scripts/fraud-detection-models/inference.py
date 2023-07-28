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
"""Usage example:
python inference.py --training-data ../../datasets/training-data/fraud-detection-training-data.csv\
        --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv\
        --model-dir ../../fraud-detection-models\
        --output-file out.txt --model-type HinSAGE
"""

import os
import pickle

import click
import dgl
import numpy as np
import pandas as pd
import torch
from model import HeteroRGCN
from model import HinSAGE

np.random.seed(1001)
torch.manual_seed(1001)


def build_fsi_graph(train_data, col_drop):
    """Build heterograph from edglist and node index.

    Args:
        train_data (pd.DataFrame): training data for node features.
        col_drop (list): features to drop from node features.

    Returns:
       Tuple[DGLGraph, torch.tensor]: dgl graph, normalized feature tensor
    """

    edge_list = {
        ('client', 'buy', 'transaction'): (train_data['client_node'].values, train_data['index'].values),
        ('transaction', 'bought', 'client'): (train_data['index'].values, train_data['client_node'].values),
        ('transaction', 'issued', 'merchant'): (train_data['index'].values, train_data['merchant_node'].values),
        ('merchant', 'sell', 'transaction'): (train_data['merchant_node'].values, train_data['index'].values)
    }

    graph = dgl.heterograph(edge_list)
    feature_tensors = torch.from_numpy(train_data.drop(col_drop, axis=1).values).float()
    feature_tensors = (feature_tensors - feature_tensors.mean(0)) / (0.0001 + feature_tensors.std(0))

    return graph, feature_tensors


def map_node_id(df, col_name):
    """ Convert column node list to integer index for dgl graph.

    Args:
        df (pd.DataFrame): dataframe
        col_name (list) : column list
    """
    node_index = {j: i for i, j in enumerate(df[col_name].unique())}
    df[col_name] = df[col_name].map(node_index)


def prepare_data(training_data, test_data):
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
     tuple of (training_data, test_data, train_index, test_index, label, combined data)
    """

    df_train = pd.read_csv(training_data)
    train_idx_ = df_train.shape[0]
    df_test = pd.read_csv(test_data)
    df = pd.concat([df_train, df_test], axis=0)
    df['tran_id'] = df['index']

    meta_cols = ['tran_id', 'client_node', 'merchant_node']
    for col in meta_cols:
        map_node_id(df, col)

    test_idx = df['tran_id'][train_idx_:]

    df['index'] = df['tran_id']
    df.index = df['index']

    return test_idx, df


def load_model(model_dir,  gnn_model=HeteroRGCN):
    """Load trained models from model directory

    Parameters
    ----------
    model_dir : str
        models directory path
    gnn_model: nn.Module
        GNN model type to load either HinSAGE or HeteroRGCN

    Returns
    -------
    (nn.Module, dgl.DGLHeteroGraph, dict)
        model, training graph, hyperparameter
    """

    from cuml import ForestInference

    with open(os.path.join(model_dir, "graph.pkl"), 'rb') as f:
        graph = pickle.load(f)
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparameters = pickle.load(f)
    model = gnn_model(graph,
                      in_size=hyperparameters['in_size'],
                      hidden_size=hyperparameters['hidden_size'],
                      out_size=hyperparameters['out_size'],
                      n_layers=hyperparameters['n_layers'],
                      embedding_size=hyperparameters['embedding_size'],
                      target=hyperparameters['target_node'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    xgb_model = ForestInference.load(os.path.join(model_dir, 'xgb.pt'), output_class=True)

    return model, xgb_model, graph


@torch.no_grad()
def evaluate(model, eval_loader, feature_tensors, target_node, device='cpu'):
    """Evaluate the specified model on the given evaluation input graph

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be evaluated.
    eval_loader : dgl.dataloading.DataLoader
        DataLoader containing the evaluation dataset.
    feature_tensors : torch.Tensor
        The feature tensors corresponding to the evaluation data.
        Shape: (num_samples, num_features).
    target_node : str
        The target node for evaluation, indicating the node of interest.
    device : str, optional
        The device where the model and tensors should be loaded.
        Default is 'cpu'.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        A tuple containing numpy arrays of logits, eval seed and embeddings
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

def inference(model, input_graph, feature_tensors, test_idx, target_node, device):

    """ Perform inference on a given model using the provided input graph and feature tensors.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be used for inference.
    input_graph : dgl.DGLHeteroGraph
        The input heterogeneous graph in DGL format. It represents the graph structure.
    feature_tensors : torch.Tensor
        The input feature tensors for nodes in the input graph. Each row corresponds to the features of a single node.
    test_idx : torch.Tensor
        The indices of the nodes in the input graph that are used for testing and evaluation.
    target_node : str, optional (default: "transaction")
        The type of node for which inference will be performed. By default, it is set to "transaction".
    device : str or torch.device, optional (default: "cuda:0" if torch.cuda.is_available() else "cpu")
        The device where the computation will take place. By default, it uses GPU ("cuda:0") if available,
          otherwise CPU ("cpu").
    Returns
    -------
    (logits, seeds, embedding) : torch.Tensor
        The logits, node seeds, embeddings using model inference.
    """

    full_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=[4, 3])
    test_dataloader = dgl.dataloading.DataLoader(input_graph, {target_node: test_idx},
                                                 full_sampler,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0)
    test_logits, test_seeds, test_embedding = evaluate(
        model, test_dataloader, feature_tensors,  target_node, device=device)

    return test_logits, test_seeds, test_embedding


@click.command()
@click.option('--training-data', help="Path to training data for graph structure.", default="data/training.csv")
@click.option('--validation-data', help="Path to validation data", default="data/validation.csv")
@click.option('--model-dir', help="path to model directory", default="modeldir")
@click.option('--target-node', help="Target node", default="transaction")
@click.option('--output-file', help="Path to csv inference result", default="out.csv")
@click.option('--model-type', help="Model type either RGCN/Graphsage", default="RGCN")
def main(training_data, validation_data, model_dir, target_node, output_file, model_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_cols = ["client_node", "merchant_node", "fraud_label", "index", "tran_id"]
    if model_type == "RGCN":
        gnn_model = HeteroRGCN
    else:
        gnn_model = HinSAGE

    # prepare data
    test_idx, all_data = prepare_data(training_data, validation_data)

    # build graph structure
    g_test, feature_tensors = build_fsi_graph(all_data, meta_cols)

    # Load graph model
    model, xgb_model, _ = load_model(model_dir, gnn_model=gnn_model)
    model = model.to(device)
    g_test = g_test.to(device)
    feature_tensors = feature_tensors.to(device)
    test_idx = torch.from_numpy(test_idx.values).to(device)

    _, test_seeds, test_embedding = inference(model, g_test, feature_tensors, test_idx, target_node, device)

    # collect result
    pred_score = xgb_model.predict_proba(test_embedding)[:, 1]
    df_result = pd.DataFrame(test_seeds.cpu(), columns=['node_id'])
    df_result['score'] = pred_score.get()

    df_result.to_csv(output_file, index=False)

    print(df_result)


if __name__ == '__main__':
    main()
