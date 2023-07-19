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
"""
python training.py --training-data ../../datasets/training-data/fraud-detection-training-data.csv\
      --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv \
          --model-dir models-dir --output-file out.txt\
            --model-type HinSAGE
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from torch import nn
from torchmetrics.functional import accuracy
from tqdm import trange
from xgboost import XGBClassifier

np.random.seed(1001)
torch.manual_seed(1001)


def get_metrics(pred, labels, name='RGCN'):
    """Compute evaluation metrics

    Args:
        pred (np.array) : prediction
        labels (np.array): groundtruth label
        name (str, optional): model name. Defaults to 'RGCN'.

    Returns:
        List[List]: List of metrics f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix, auc_r
    """

    pred, pred_proba = pred.argmax(1), pred[:, 1]

    acc = ((pred == labels)).sum() / len(pred)

    true_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    false_pos = (np.where(pred == 1, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()
    false_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 1, 1, 0) > 1).sum()
    true_neg = (np.where(pred == 0, 1, 0) + np.where(labels == 0, 1, 0) > 1).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    f_1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    confusion_matrix = pd.DataFrame(np.array([[true_pos, false_pos], [false_neg, true_neg]]),
                                    columns=["labels positive", "labels negative"],
                                    index=["predicted positive", "predicted negative"])

    average_precision = average_precision_score(labels, pred_proba)

    fpr, tpr, _ = roc_curve(labels, pred_proba)
    prc, rec, _ = precision_recall_curve(labels, pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prc)
    auc_r = (fpr, tpr, roc_auc, name)
    return (acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, confusion_matrix, auc_r)


def build_fsi_graph(train_data, col_drop):
    """Build heterograph from edglist and node index.

    Args:
        train_data (pd.DataFrame): training data for node features.
        col_drop (list): features to drop from node features.

    Returns:
       Tuple[DGLGraph, torch.tensor]: dlg graph, normalized feature tensor
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

    train_idx = df['tran_id'][:train_idx_]
    test_idx = df['tran_id'][train_idx_:]

    df['index'] = df['tran_id']
    df.index = df['index']

    return (df.iloc[train_idx, :], df.iloc[test_idx, :], train_idx, test_idx, df['fraud_label'].values, df)


def save_model(graph, model, hyperparameters, xgb_model, model_dir):
    """Save trained model with graph & hyperparameters dict

    Args:
        graph (DGLHeteroGraph): dgl graph
        model (HeteroRGCN): trained RGCN model
        hyperparameters (dict): hyperparameter for model training.
        xgb_model (XGBoost): XGBoost trained model.
        model_dir (str): directory to save
    """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'wb') as f:
        pickle.dump(hyperparameters, f)
    with open(os.path.join(model_dir, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    xgb_model.save_model(os.path.join(model_dir, "xgb.pt"))


def load_model(model_dir, gnn_model=HeteroRGCN):
    """Load trained model, graph structure from given directory

    Args:
        model_dir (str path):directory path for trained model obj.

    Returns:
        List[HeteroRGCN, DGLHeteroGraph]: model and graph structure.
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


def init_loaders(g_train, train_idx, test_idx, val_idx, g_test, target_node='transaction', batch_size=100):
    """Initialize dataloader and graph sampler. For training use neighbor sampling.

    Args:
        g_train (DGLHeteroGraph): train graph
        train_idx (list): train feature index
        test_idx (list): test feature index
        val_idx (list): validation index
        g_test (DGLHeteroGraph): test graph
        target_node (str, optional): target node. Defaults to 'authentication'.
        batch_size (int): batchsize for inference.

    Returns:
        List[NodeDataLoader,NodeDataLoader,NodeDataLoader]: list of dataloaders
    """

    neighbor_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    full_sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 3])

    train_dataloader = dgl.dataloading.DataLoader(g_train, {target_node: train_idx},
                                                  neighbor_sampler,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  use_uva=False)

    test_dataloader = dgl.dataloading.DataLoader(g_test, {target_node: test_idx},
                                                 full_sampler,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=0,
                                                 use_uva=False)

    val_dataloader = dgl.dataloading.DataLoader(g_test, {target_node: val_idx},
                                                neighbor_sampler,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0,
                                                use_uva=False)

    return train_dataloader, val_dataloader, test_dataloader


def train(model,
          loss_func,
          train_dataloader,
          labels,
          optimizer,
          feature_tensors,
          target_node='transaction',
          device='cpu'):
    """Train RGCN model

    Args:
        model(HeteroRGCN): RGCN model
        loss_func (nn.loss) : loss function
        train_dataloader (NodeDataLoader) : train dataloader class
        labels (list): training label
        optimizer (nn.optimizer) : optimizer for training
        feature_tensors (torch.from_numpy) : node features
        target_node (str, optional): target node embedding. Defaults to 'transaction'.
        device (str, optional): host device. Defaults to 'cpu'.

    Returns:
        _type_: training accuracy and training loss
    """
    model.train()
    train_loss = 0.0
    for _, (_, output_nodes, blocks) in enumerate(train_dataloader):
        seed = output_nodes[target_node]
        blocks = [b.to(device) for b in blocks]
        nid = blocks[0].srcnodes[target_node].data[dgl.NID]
        input_features = feature_tensors[nid].to(device)

        logits, _ = model(blocks, input_features)
        loss = loss_func(logits, labels[seed])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc = accuracy_score(logits.argmax(1).cpu(), labels[seed].cpu().long()).item()

    return train_acc, train_loss


@torch.no_grad()
def evaluate(model, eval_loader, feature_tensors, target_node, device='cpu'):
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


@click.command()
@click.option('--training-data', help="Path to training data ", default="data/training.csv")
@click.option('--validation-data', help="Path to validation data", default="data/validation.csv")
@click.option('--model-dir', help="path to model directory", default="debug")
@click.option('--target-node', help="Target node", default="transaction")
@click.option('--epochs', help="Number of epochs", default=20)
@click.option('--batch_size', help="Batch size", default=1024)
@click.option('--output-file', help="Path to csv inference result", default="debug/out.csv")
@click.option('--model-type', help="Model type either RGCN/Graphsage", default="RGCN")
def train_model(training_data, validation_data, model_dir, target_node, epochs, batch_size, output_file, model_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == "RGCN":
        gnn_model = HeteroRGCN
    else:
        gnn_model = HinSAGE

    # process training data
    train_data, _, train_idx, inductive_idx,\
        labels, df = prepare_data(training_data, validation_data)
    meta_cols = ["client_node", "merchant_node", "fraud_label", "index", "tran_id"]

    # Build graph
    whole_graph, feature_tensors = build_fsi_graph(df, meta_cols)
    train_graph, _ = build_fsi_graph(train_data, meta_cols)
    whole_graph = whole_graph.to(device)

    feature_tensors = feature_tensors.to(device)
    train_idx = torch.from_numpy(train_idx.values).to(device)
    inductive_idx = torch.from_numpy(inductive_idx.values).to(device)
    labels = torch.LongTensor(labels).to(device)

    # Hyperparameters
    in_size, hidden_size, out_size, n_layers,\
        embedding_size = 111, 64, 2, 2, 1
    hyperparameters = {
        "in_size": in_size,
        "hidden_size": hidden_size,
        "out_size": out_size,
        "n_layers": n_layers,
        "embedding_size": embedding_size,
        "target_node": target_node,
        "epoch": epochs
    }

    scale_pos_weight = train_data['fraud_label'].sum() / train_data.shape[0]
    scale_pos_weight = torch.FloatTensor([scale_pos_weight, 1 - scale_pos_weight]).to(device)

    # Dataloaders
    train_loader, val_loader, test_loader = init_loaders(train_graph.to(
        device), train_idx, test_idx=inductive_idx,
        val_idx=inductive_idx, g_test=whole_graph, batch_size=batch_size)

    # Set model variables
    # Set model variables
    model = gnn_model(whole_graph, in_size, hidden_size, out_size, n_layers, embedding_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_func = nn.CrossEntropyLoss(weight=scale_pos_weight.float())

    for epoch in trange(epochs):

        train_acc, loss = train(
            model, loss_func, train_loader, labels, optimizer, feature_tensors,
            target_node, device=device)
        print(f"Epoch {epoch}/{epochs} | Train Accuracy: {train_acc} | Train Loss: {loss}")
        val_logits, val_seed, _ = evaluate(model, val_loader, feature_tensors, target_node, device=device)
        val_accuracy = accuracy(val_logits.argmax(1), labels.long()[val_seed].cpu(), "binary").item()
        val_auc = roc_auc_score(
            labels.long()[val_seed].cpu().numpy(),
            val_logits[:, 1].numpy(),
        )
        print(f"Validation Accuracy: {val_accuracy} auc {val_auc}")

    # Create embeddings
    _, train_seeds, train_embedding = evaluate(model, train_loader, feature_tensors, target_node, device=device)
    test_logits, test_seeds, test_embedding = evaluate(model, test_loader, feature_tensors, target_node, device=device)

    # compute metrics
    test_acc = accuracy(test_logits.argmax(dim=1), labels.long()[test_seeds].cpu(), "binary").item()
    test_auc = roc_auc_score(labels.long()[test_seeds].cpu().numpy(), test_logits[:, 1].numpy())

    metrics_result = pd.DataFrame()
    print(f"Final Test Accuracy: {test_acc} auc {test_auc}")
    acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, _, _ = get_metrics(
        test_logits.numpy(), labels[test_seeds].cpu().numpy())
    metrics_result = [{
        'model': 'RGCN',
        'acc': acc,
        'f1': f_1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': average_precision
    }]

    # Train XGBoost classifier on embedding vector
    classifier = XGBClassifier(n_estimators=100)
    classifier.fit(train_embedding.cpu().numpy(), labels[train_seeds].cpu().numpy())
    xgb_pred = classifier.predict_proba(test_embedding.cpu().numpy())
    acc, f_1, precision, recall, roc_auc, pr_auc, average_precision, _, _ = get_metrics(
        xgb_pred, labels[inductive_idx].cpu().numpy(),  name='RGCN_XGB')
    metrics_result += [{
        'model': 'RGCN_XGB',
        'acc': acc,
        'f1': f_1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': average_precision
    }]

    # Save model
    pd.DataFrame(metrics_result).to_csv(output_file)
    save_model(whole_graph, model, hyperparameters, classifier, model_dir)


if __name__ == "__main__":
    train_model()
