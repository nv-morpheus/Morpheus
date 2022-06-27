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
""""
# EXample usage:
python training.py --training-data ../../datasets/training-data/fraud-detection-training-data.csv \
     --validation-data ../../datasets/validation-data/fraud-detection-validation-data.csv \
         --epoch 10 --output-xgb model/xgb.pt --output-hinsage model/hinsage.pt
"""

import argparse
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation import Evaluation
from stellargraph import StellarGraph
from stellargraph.layer import HinSAGE
from stellargraph.mapper import HinSAGENodeGenerator
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy
from xgboost import XGBClassifier

tf.random.set_seed(1001)


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

    transaction_node_data = dataset.drop(["client_node", "merchant_node", "fraud_label", "index"], axis=1)
    client_node_data = pd.DataFrame([1] * len(dataset.client_node.unique())).set_index(dataset.client_node.unique())
    merchant_node_data = pd.DataFrame([1] * len(dataset.merchant_node.unique())).set_index(
        dataset.merchant_node.unique())

    nodes = {"client": dataset.client_node, "merchant": dataset.merchant_node, "transaction": dataset.index}
    edges = [zip(dataset.client_node, dataset.index), zip(dataset.merchant_node, dataset.index)]
    features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}
    graph = graph_construction(nodes, edges, features)

    return graph


def split_train_test(df, ratio=0.7, train_anom_prop=0.1, test_anom_prop=0.1):
    cutoff = round(ratio * len(df))
    train_data = df.head(cutoff)
    test_data = df.tail(len(df) - cutoff)

    train_fraud = np.random.choice(train_data[train_data.fraud_label == 1].index,
                                   int((1 - train_anom_prop) * train_data.shape[0]))
    test_fraud = np.random.choice(test_data[test_data.fraud_label == 1].index,
                                  int((1 - test_anom_prop) * test_data.shape[0]))

    train_data, test_data = train_data[~train_data.index.isin(
        train_fraud)], test_data[~test_data.index.isin(test_fraud)]
    return train_data, test_data, train_data.index, test_data.index


def data_preprocessing(training_dataset):

    # Load dataset
    df = pd.read_csv(training_dataset)
    train_data, test_data, train_data_index, test_data_index = split_train_test(df, 0.7, 1.0, 0.7)
    return train_data, test_data, train_data_index, test_data_index


def train_model(train_graph, node_identifiers, label):
    # train_graph: Stellar graph structure.
    # Train graphsage and GBT model.

    # Global parameters:
    batch_size = 5
    xgb_n_estimator = 100
    num_samples = [2, 32]

    # The mapper feeds data from sampled subgraph to GraphSAGE model
    train_node_identifiers = node_identifiers[:round(0.8 * len(node_identifiers))]
    train_labels = label.loc[train_node_identifiers]

    validation_node_identifiers = node_identifiers[round(0.8 * len(node_identifiers)):]
    validation_labels = label.loc[validation_node_identifiers]
    generator = HinSAGENodeGenerator(train_graph, batch_size, num_samples, head_node_type=embedding_node_type)
    train_gen = generator.flow(train_node_identifiers, train_labels, shuffle=True)
    test_gen = generator.flow(validation_node_identifiers, validation_labels)

    # HinSAGE model
    model = HinSAGE(layer_sizes=[embedding_size] * len(num_samples), generator=generator, dropout=0)
    x_inp, x_out = model.build()

    # Final estimator layer
    prediction = layers.Dense(units=1, activation="sigmoid", dtype='float32')(x_out)

    # Create Keras model for training
    model = Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=binary_crossentropy,
    )

    # Train Model
    model.fit(train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=False)

    hinsage_model = Model(inputs=x_inp, outputs=x_out)
    train_gen_not_shuffled = generator.flow(node_identifiers, label, shuffle=False)
    embeddings_train = hinsage_model.predict(train_gen_not_shuffled)

    inductive_embedding = pd.DataFrame(embeddings_train, index=node_identifiers)

    xgb_model = XGBClassifier(n_estimators=xgb_n_estimator)
    xgb_model.fit(inductive_embedding, label)

    return {"hinsage": hinsage_model, "xgb": xgb_model}


def save_model(model, output_xgboost, output_hinsage):
    # model: dict of xgb & hsg model
    # Save as tensorflow model file

    model['hinsage'].save(output_hinsage)
    model['xgb'].save_model(output_xgboost)


def inductive_step_hinsage(S, trained_model, inductive_node_identifiers, batch_size):
    """

    This function generates embeddings for unseen nodes using a trained hinsage model.
    It returns the embeddings for these unseen nodes.

    Parameters
    ----------
    S : StellarGraph Object
        The graph on which HinSAGE is deployed.
    trained_model : Neural Network
        The trained hinsage model, containing the trained and optimized aggregation functions per depth.
    inductive_node_identifiers : list
        Defines the nodes that HinSAGE needs to generate embeddings for
    batch_size: int
        batch size for the neural network in which HinSAGE is implemented.

    """

    # The mapper feeds data from sampled subgraph to HinSAGE model
    generator = HinSAGENodeGenerator(S, batch_size, num_samples, head_node_type="transaction")
    test_gen_not_shuffled = generator.flow(inductive_node_identifiers, shuffle=False)

    inductive_emb = trained_model.predict(test_gen_not_shuffled, verbose=1)
    inductive_emb = pd.DataFrame(inductive_emb, index=inductive_node_identifiers)

    return inductive_emb


def model_eval(trained_model, S, node_identifier, label):

    inductive_emb = inductive_step_hinsage(S, trained_model['hinsage'], node_identifier, batch_size=5)
    predictions = trained_model['xgb'].predict_proba(inductive_emb)
    # evaluate performance.
    eval = Evaluation(predictions, label, "GraphSAGE+features")
    eval.f1_ap_rec()
    print(f"AUC -- {eval.roc_curve()}")


def main():
    print("Data Preprocessing...")
    train_data = pd.read_csv(args.training_data)

    val_data = pd.read_csv(args.validation_data)
    val_data.index = val_data['index']
    # train_data, val_data, train_data_index, val_data_index = split_train_test(df, 0.7, 1.0,0.7)

    print("Graph construction")
    S_graph = build_graph_features(train_data)
    print("Model Training...")
    model = train_model(S_graph, node_identifiers=list(train_data.index), label=train_data['fraud_label'])
    # print(model)
    print("Save trained model")
    if args.save_model:
        save_model(model, args.output_xgb, args.output_hinsage)
    # Save graph info
    print("Model Evaluation...")
    inductive_data = pd.concat((train_data, val_data))
    S_graph = build_graph_features(inductive_data)
    model_eval(model, S_graph, node_identifier=list(val_data.index), label=val_data['fraud_label'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", required=True, help="CSV with fraud_label")
    parser.add_argument("--validation-data", required=False, help="CSV with fraud_label")
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--node_type", required=False, help="Target node type", default="transaction")
    parser.add_argument("--output-xgb", required=False, help="output file to save xgboost model")
    parser.add_argument("--output-hinsage", required=False, help="output file to save GraphHinSage model")
    parser.add_argument("--save_model", type=bool, default=False, help="Save models to give filenames")
    parser.add_argument("--embedding_size", required=False, default=64, help="output file to save new model")

    args = parser.parse_args()

    # Global parameters:
    embedding_size = int(args.embedding_size)
    epochs = int(args.epochs)
    embedding_node_type = str(args.node_type)
    num_samples = [2, 32]

    main()