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

"""
Example Usage:
python inference.py --graph-data ../../datasets/training-data/fraud-detection-training-data.csv \
     --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv \
     --model-xgb model/xgb-model.pt --model-hinsage model/hinsage-model.pt --output out.txt
"""

import argparse

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import f1_score
from stellargraph import StellarGraph
from stellargraph.layer import HinSAGE
from stellargraph.mapper import HinSAGENodeGenerator
import cudf
from cuml import ForestInference


def graph_construction(nodes, edges, node_features):
    g_nx = nx.Graph()

    # add nodes
    for key, values in nodes.items():
        g_nx.add_nodes_from(values, ntype=key)
    # add edges
    for edge in edges:
        g_nx.add_edges_from(edge)

    return StellarGraph(g_nx, node_type_name="ntype", node_features=node_features)

def build_graph_features(dataset):
    # Train data
    transaction_node_data = dataset.drop(["client_node","merchant_node","fraud_label", "index"], axis=1)
    client_node_data = pd.DataFrame([1]*len(dataset.client_node.unique())).set_index(dataset.client_node.unique())
    merchant_node_data = pd.DataFrame([1]*len(dataset.merchant_node.unique())).set_index(dataset.merchant_node.unique())

    nodes = {"client":dataset.client_node, "merchant":dataset.merchant_node, "transaction":dataset.index}
    edges = [zip(dataset.client_node, dataset.index),zip(dataset.merchant_node, dataset.index)]
    features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

    graph = graph_construction(nodes, edges, features) #GraphConstruction(nodes, edges, features)
    #S = graph.get_stellargraph()
    return graph


def inductive_step_hinsage(S, trained_model, inductive_node_identifiers, batch_size):
    # perform inductive learning from trained graph model

    num_samples = [2,32]
    # The mapper feeds data from sampled subgraph to HinSAGE model
    generator = HinSAGENodeGenerator(S, batch_size, num_samples, head_node_type="transaction")
    test_gen_not_shuffled = generator.flow(inductive_node_identifiers, shuffle=False )

    inductive_emb = trained_model.predict(test_gen_not_shuffled, verbose=1)
    inductive_emb = pd.DataFrame(inductive_emb, index=inductive_node_identifiers)

    return inductive_emb

def infer(model_xgb, model_hinsage, graph_data, node_identifier, output):

    # Build graph structure.

    graph = build_graph_features(graph_data)


    # Load XGBoost & GraphSAGE model
    xgb_model = ForestInference.load(model_xgb, output_class=True)
    hgs_model = tf.keras.models.load_model(model_hinsage)

    inductive_embedding = inductive_step_hinsage(graph, hgs_model, node_identifier, batch_size=5)

    # prediction
    prediction = xgb_model.predict_proba(inductive_embedding)[:,1]
    result = pd.DataFrame(node_identifier, columns=['node_id'])
    result['prediction'] = prediction
    result.to_csv(output, index=False)
    return result


def main():
    graph_data = pd.read_csv(args.graph_data)
    val_data = pd.read_csv(args.validation_data)
    graph_data = pd.concat([graph_data, val_data])
    graph_data = graph_data.set_index(graph_data['index'])

    infer(args.model_xgb, args.model_hinsage, graph_data=graph_data, node_identifier=list(val_data['index']), output=args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-data", required=False,help="Labelled data in CSV format")
    parser.add_argument("--model-hinsage", required=True, help="trained hinsage model")
    parser.add_argument("--model-xgb", required=True, help="trained xgb model")
    parser.add_argument("--graph-data", help="Training dataset for graph structure", required=True)
    parser.add_argument("--output", required=True, help="output filename")
    args = parser.parse_args()

main()
