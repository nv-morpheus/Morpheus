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

import pandas as pd
from stellargraph.layer import HinSAGE
from stellargraph.mapper import HinSAGENodeGenerator
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy


class HinSAGE_Representation_Learner:
    """
    This class initializes a graphsage framework

    Parameters
    ----------
    embedding_size : int
        The desired size of the resulting embeddings
    num_samples : list
        The length of the list defines the depth of random walks, the values of the list
        define the number of nodes to sample per neighborhood.
    embedding_for_node_type: str
        String identifying the node type for which we want graphsage to generate embeddings.

    """

    def __init__(self, embedding_size, num_samples, embedding_for_node_type):

        self.embedding_size = embedding_size
        self.num_samples = num_samples
        self.embedding_for_node_type = embedding_for_node_type

    def train_hinsage(self, S, node_identifiers, label, batch_size, epochs):
        """

        This function trains a HinSAGE model, implemented in Tensorflow.
        It returns the trained HinSAGE model and a pandas dataframe
        containing the embeddings generated for the train nodes.

        Parameters
        ----------
        S : StellarGraph Object
            The graph on which HinSAGE trains its aggregator functions.
        node_identifiers : list
            Defines the nodes that HinSAGE uses to train its aggregation functions.
        label: Pandas dataframe
            Defines the label of the nodes used for training, with the index representing the nodes.
        batch_size: int
            batch size to train the neural network in which HinSAGE is implemented.
        epochs: int
            Number of epochs for the neural network.

        """
        # The mapper feeds data from sampled subgraph to GraphSAGE model
        train_node_identifiers = node_identifiers[:round(0.8 * len(node_identifiers))]
        train_labels = label.loc[train_node_identifiers]

        validation_node_identifiers = node_identifiers[round(0.8 * len(node_identifiers)):]
        validation_labels = label.loc[validation_node_identifiers]
        generator = HinSAGENodeGenerator(S, batch_size, self.num_samples, head_node_type=self.embedding_for_node_type)
        train_gen = generator.flow(train_node_identifiers, train_labels, shuffle=True)
        test_gen = generator.flow(validation_node_identifiers, validation_labels)

        # HinSAGE model
        model = HinSAGE(layer_sizes=[self.embedding_size] * len(self.num_samples), generator=generator, dropout=0)
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

        trained_model = Model(inputs=x_inp, outputs=x_out)
        train_gen_not_shuffled = generator.flow(node_identifiers, label, shuffle=False)
        embeddings_train = trained_model.predict(train_gen_not_shuffled)

        train_emb = pd.DataFrame(embeddings_train, index=node_identifiers)

        return trained_model, train_emb

    def inductive_step_hinsage(self, S, trained_model, inductive_node_identifiers, batch_size):
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
        generator = HinSAGENodeGenerator(S, batch_size, self.num_samples, head_node_type=self.embedding_for_node_type)
        test_gen_not_shuffled = generator.flow(inductive_node_identifiers, shuffle=False)

        inductive_emb = trained_model.predict(test_gen_not_shuffled, verbose=1)
        inductive_emb = pd.DataFrame(inductive_emb, index=inductive_node_identifiers)

        return inductive_emb
