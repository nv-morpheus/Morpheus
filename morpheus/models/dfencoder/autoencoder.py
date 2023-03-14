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

# Original Source: https:#github.com/AlliedToasters/dfencoder
#
# Original License: BSD-3-Clause license, included below

# Copyright (c) 2019, Michael Klear.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the dfencoder Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gc
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import tqdm

from .dataframe import EncoderDataFrame
from .logging import BasicLogger, IpynbLogger, TensorboardXLogger
from .scalers import GaussRankScaler, NullScaler, StandardScaler, ModifiedScaler


def ohe(input_vector, dim, device="cpu"):
    """Does one-hot encoding of input vector."""
    batch_size = len(input_vector)
    nb_digits = dim

    y = input_vector.reshape(-1, 1)
    y_onehot = torch.FloatTensor(batch_size, nb_digits).to(device)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


def compute_embedding_size(n_categories):
    """
    Applies a standard formula to choose the number of feature embeddings
    to use in a given embedding layers.

    n_categories is the number of unique categories in a column.
    """
    val = min(600, round(1.6 * n_categories**0.56))
    return int(val)


class CompleteLayer(torch.nn.Module):
    """
    Impliments a layer with linear transformation
    and optional activation and dropout."""

    def __init__(self, in_dim, out_dim, activation=None, dropout=None, *args, **kwargs):
        super(CompleteLayer, self).__init__(*args, **kwargs)
        self.layers = []
        linear = torch.nn.Linear(in_dim, out_dim)
        self.layers.append(linear)
        self.add_module('linear_layer', linear)

        self.activation = activation
        if activation is not None:
            act = self.interpret_activation(activation)
            self.layers.append(act)
        if dropout is not None:
            dropout_layer = torch.nn.Dropout(dropout)
            self.layers.append(dropout_layer)
            self.add_module('dropout', dropout_layer)

    def interpret_activation(self, act=None):
        if act is None:
            act = self.activation
        activations = {
            'leaky_relu': torch.nn.functional.leaky_relu,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'selu': torch.selu,
            'hardtanh': torch.nn.functional.hardtanh,
            'relu6': torch.nn.functional.relu6,
            'elu': torch.nn.functional.elu,
            'celu': torch.nn.functional.celu,
            'rrelu': torch.nn.functional.rrelu,
            'hardshrink': torch.nn.functional.hardshrink,
            'tanhshrink': torch.nn.functional.tanhshrink,
            'softsign': torch.nn.functional.softsign
        }
        try:
            return activations[act]
        except:
            msg = f'activation {act} not understood. \n'
            msg += 'please use one of: \n'
            msg += str(list(activations.keys()))
            raise Exception(msg)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AutoEncoder(torch.nn.Module):

    def __init__(
            self,
            encoder_layers=None,
            decoder_layers=None,
            encoder_dropout=None,
            decoder_dropout=None,
            encoder_activations=None,
            decoder_activations=None,
            activation='relu',
            min_cats=10,
            swap_p=.15,
            lr=0.01,
            batch_size=256,
            eval_batch_size=1024,
            optimizer='adam',
            amsgrad=False,
            momentum=0,
            betas=(0.9, 0.999),
            dampening=0,
            weight_decay=0,
            lr_decay=None,
            nesterov=False,
            verbose=False,
            device=None,
            logger='basic',
            logdir='logdir/',
            project_embeddings=True,
            run=None,
            progress_bar=True,
            n_megabatches=1,
            scaler='standard',
            patience=5,
            preset_cats=None,
            loss_scaler='standard',  # scaler for the losses (z score)
            *args,
            **kwargs):
        super(AutoEncoder, self).__init__(*args, **kwargs)
        self.numeric_fts = OrderedDict()
        self.binary_fts = OrderedDict()
        self.categorical_fts = OrderedDict()
        self.cyclical_fts = OrderedDict()
        self.feature_loss_stats = dict()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_activations = encoder_activations
        self.decoder_activations = decoder_activations
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.min_cats = min_cats
        self.preset_cats = preset_cats
        self.encoder = []
        self.decoder = []
        self.train_mode = self.train

        self.swap_p = swap_p
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.numeric_output = None
        self.binary_output = None

        # `num_names` is a list of column names that contain numeric data (int & float fields).
        self.num_names = []
        self.bin_names = []

        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.amsgrad = amsgrad
        self.momentum = momentum
        self.betas = betas
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.optim = None
        self.progress_bar = progress_bar

        self.mse = torch.nn.modules.loss.MSELoss(reduction='none')
        self.bce = torch.nn.modules.loss.BCELoss(reduction='none')
        self.cce = torch.nn.modules.loss.CrossEntropyLoss(reduction='none')

        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.logger = logger
        self.logdir = logdir
        self.run = run
        self.project_embeddings = project_embeddings
        self.scaler = scaler
        self.patience = patience

        # scaler class used to scale losses and collect loss stats
        self.loss_scaler_str = loss_scaler
        self.loss_scaler = self.get_scaler(loss_scaler)

        self.n_megabatches = n_megabatches

    def get_scaler(self, name):
        scalers = {
            'standard': StandardScaler,
            'gauss_rank': GaussRankScaler,
            'modified': ModifiedScaler,
            None: NullScaler,
            'none': NullScaler
        }
        return scalers[name]

    def init_numeric(self, df):
        dt = df.dtypes
        numeric = []
        numeric += list(dt[dt == int].index)
        numeric += list(dt[dt == float].index)

        if isinstance(self.scaler, str):
            scalers = {ft: self.scaler for ft in numeric}
        elif isinstance(self.scaler, dict):
            scalers = self.scaler

        for ft in numeric:
            Scaler = self.get_scaler(scalers.get(ft, 'gauss_rank'))
            feature = {'mean': df[ft].mean(), 'std': df[ft].std(), 'scaler': Scaler()}
            feature['scaler'].fit(df[ft][~df[ft].isna()].values)
            self.numeric_fts[ft] = feature

        self.num_names = list(self.numeric_fts.keys())

    def create_numerical_col_max(self, num_names, mse_loss):
        if num_names:
            num_df = pd.DataFrame(num_names)
            num_df.columns = ['num_col_max_loss']
            num_df.reset_index(inplace=True)
            argmax_df = pd.DataFrame(torch.argmax(mse_loss.cpu(), dim=1).numpy())
            argmax_df.columns = ['index']
            num_df = num_df.merge(argmax_df, on='index', how='left')
            num_df.drop('index', axis=1, inplace=True)
        else:
            num_df = pd.DataFrame()
        return num_df

    def create_binary_col_max(self, bin_names, bce_loss):
        if bin_names:
            bool_df = pd.DataFrame(bin_names)
            bool_df.columns = ['bin_col_max_loss']
            bool_df.reset_index(inplace=True)
            argmax_df = pd.DataFrame(torch.argmax(bce_loss.cpu(), dim=1).numpy())
            argmax_df.columns = ['index']
            bool_df = bool_df.merge(argmax_df, on='index', how='left')
            bool_df.drop('index', axis=1, inplace=True)
        else:
            bool_df = pd.DataFrame()
        return bool_df

    def create_categorical_col_max(self, cat_names, cce_loss):
        final_list = []
        if cat_names:
            for index, val in enumerate(cce_loss):
                val = pd.DataFrame(val.cpu().numpy())
                val.columns = [cat_names[index]]
                final_list.append(val)
            cat_df = pd.DataFrame(pd.concat(final_list, axis=1).idxmax(axis=1))
            cat_df.columns = ['cat_col_max_loss']
        else:
            cat_df = pd.DataFrame()
        return cat_df

    def get_variable_importance(self, num_names, cat_names, bin_names, mse_loss, bce_loss, cce_loss, cloudtrail_df):
        # Get data in the right format
        num_df = self.create_numerical_col_max(num_names, mse_loss)
        bool_df = self.create_binary_col_max(bin_names, bce_loss)
        cat_df = self.create_categorical_col_max(cat_names, cce_loss)
        variable_importance_df = pd.concat([num_df, bool_df, cat_df], axis=1)
        return variable_importance_df

    def return_feature_names(self):
        bin_names = list(self.binary_fts.keys())
        num_names = list(self.numeric_fts.keys())
        cat_names = list(self.categorical_fts.keys())
        return num_names, cat_names, bin_names

    def init_cats(self, df):
        dt = df.dtypes
        objects = list(dt[dt == "object"].index)
        for ft in objects:
            feature = {}
            vl = df[ft].value_counts()
            cats = list(vl[vl >= self.min_cats].index)
            feature['cats'] = cats
            self.categorical_fts[ft] = feature

    def init_binary(self, df):
        dt = df.dtypes
        binaries = list(dt[dt == bool].index)
        for ft in self.binary_fts:
            feature = self.binary_fts[ft]
            for i, cat in enumerate(feature['cats']):
                feature[cat] = bool(i)
        for ft in binaries:
            feature = dict()
            feature['cats'] = [True, False]
            feature[True] = True
            feature[False] = False
            self.binary_fts[ft] = feature

        self.bin_names = list(self.binary_fts.keys())

    def init_features(self, df):
        if self.preset_cats is not None:
            self.categorical_fts = self.preset_cats
        else:
            self.init_cats(df)
        self.init_numeric(df)
        self.init_binary(df)

    def build_inputs(self):
        # will compute total number of inputs
        input_dim = 0

        # create categorical variable embedding layers
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            n_cats = len(feature['cats']) + 1
            embed_dim = compute_embedding_size(n_cats)
            embed_layer = torch.nn.Embedding(n_cats, embed_dim)
            feature['embedding'] = embed_layer
            self.add_module(f'{ft} embedding', embed_layer)
            # track embedding inputs
            input_dim += embed_dim

        # include numeric and binary fts
        input_dim += len(self.numeric_fts)
        input_dim += len(self.binary_fts)

        return input_dim

    def build_outputs(self, dim):
        self.numeric_output = torch.nn.Linear(dim, len(self.numeric_fts))
        self.binary_output = torch.nn.Linear(dim, len(self.binary_fts))

        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            cats = feature['cats']
            layer = torch.nn.Linear(dim, len(cats) + 1)
            feature['output_layer'] = layer
            self.add_module(f'{ft} output', layer)

    def prepare_df(self, df):
        """
        Does data preparation on copy of input dataframe.
        Returns copy.
        """
        output_df = EncoderDataFrame()
        for ft in self.numeric_fts:
            feature = self.numeric_fts[ft]
            col = df[ft].fillna(feature['mean'])
            trans_col = feature['scaler'].transform(col.values)
            trans_col = pd.Series(index=df.index, data=trans_col)
            output_df[ft] = trans_col

        for ft in self.binary_fts:
            feature = self.binary_fts[ft]
            output_df[ft] = df[ft].apply(lambda x: feature.get(x, False))

        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            col = pd.Categorical(df[ft], categories=feature['cats'] + ['_other'])
            col = col.fillna('_other')
            output_df[ft] = col

        return output_df

    def build_optimizer(self):

        lr = self.lr
        params = self.parameters()
        if self.optimizer == 'adam':
            return torch.optim.Adam(params,
                                    lr=self.lr,
                                    amsgrad=self.amsgrad,
                                    weight_decay=self.weight_decay,
                                    betas=self.betas)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
            )

    def build_model(self, df):
        """
        Takes a pandas dataframe as input.
        Builds autoencoder model.

        Returns the dataframe after making changes.
        """
        if self.verbose:
            print('Building model...')

        # get metadata from features
        self.init_features(df)
        input_dim = self.build_inputs()

        # construct a canned denoising autoencoder architecture
        if self.encoder_layers is None:
            self.encoder_layers = [int(4 * input_dim) for _ in range(3)]

        if self.decoder_layers is None:
            self.decoder_layers = []

        if self.encoder_activations is None:
            self.encoder_activations = [self.activation for _ in self.encoder_layers]

        if self.decoder_activations is None:
            self.decoder_activations = [self.activation for _ in self.decoder_layers]

        if self.encoder_dropout is None or type(self.encoder_dropout) == float:
            drp = self.encoder_dropout
            self.encoder_dropout = [drp for _ in self.encoder_layers]

        if self.decoder_dropout is None or type(self.decoder_dropout) == float:
            drp = self.decoder_dropout
            self.decoder_dropout = [drp for _ in self.decoder_layers]

        for i, dim in enumerate(self.encoder_layers):
            activation = self.encoder_activations[i]
            layer = CompleteLayer(input_dim, dim, activation=activation, dropout=self.encoder_dropout[i])
            input_dim = dim
            self.encoder.append(layer)
            self.add_module(f'encoder_{i}', layer)

        for i, dim in enumerate(self.decoder_layers):
            activation = self.decoder_activations[i]
            layer = CompleteLayer(input_dim, dim, activation=activation, dropout=self.decoder_dropout[i])
            input_dim = dim
            self.decoder.append(layer)
            self.add_module(f'decoder_{i}', layer)

        # set up predictive outputs
        self.build_outputs(dim)

        # get optimizer
        self.optim = self.build_optimizer()
        if self.lr_decay is not None:
            self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optim, self.lr_decay)

        cat_names = list(self.categorical_fts.keys())
        fts = self.num_names + self.bin_names + cat_names
        if self.logger == 'basic':
            self.logger = BasicLogger(fts=fts)
        elif self.logger == 'ipynb':
            self.logger = IpynbLogger(fts=fts)
        elif self.logger == 'tensorboard':
            self.logger = TensorboardXLogger(logdir=self.logdir, run=self.run, fts=fts)
        # returns a copy of preprocessed dataframe.
        self.to(self.device)

        if self.verbose:
            print('done!')

    def compute_targets(self, df):
        num = torch.tensor(df[self.num_names].values).float().to(self.device)
        bin = torch.tensor(df[self.bin_names].astype(int).values).float().to(self.device)
        codes = []
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            code = torch.tensor(df[ft].cat.codes.astype(int).values).to(self.device)
            codes.append(code)
        return num, bin, codes

    def encode_input(self, df):
        """
        Handles raw df inputs.
        Passes categories through embedding layers.
        """
        num, bin, codes = self.compute_targets(df)
        embeddings = []
        for i, ft in enumerate(self.categorical_fts):
            feature = self.categorical_fts[ft]
            emb = feature['embedding'](codes[i])
            embeddings.append(emb)
        return [num], [bin], embeddings

    def build_input_tensor(self, df):
        num, bin, embeddings = self.encode_input(df)
        x = torch.cat(num + bin + embeddings, dim=1)
        return x

    def compute_outputs(self, x):
        num = self.numeric_output(x)
        bin = self.binary_output(x)
        bin = torch.sigmoid(bin)
        cat = []
        for ft in self.categorical_fts:
            feature = self.categorical_fts[ft]
            out = feature['output_layer'](x)
            cat.append(out)
        return num, bin, cat

    def encode(self, x, layers=None):
        if layers is None:
            layers = len(self.encoder)
        for i in range(layers):
            layer = self.encoder[i]
            x = layer(x)
        return x

    def decode(self, x, layers=None):
        if layers is None:
            layers = len(self.decoder)
        for i in range(layers):
            layer = self.decoder[i]
            x = layer(x)
        num, bin, cat = self.compute_outputs(x)
        return num, bin, cat

    def forward(self, input):
        encoding = self.encode(input)
        num, bin, cat = self.decode(encoding)

        return num, bin, cat

    def compute_loss(self, num, bin, cat, target_df, logging=True, _id=False):
        if logging:
            if self.logger is not None:
                logging = True
            else:
                logging = False
        net_loss = []
        num_target, bin_target, codes = self.compute_targets(target_df)
        mse_loss = self.mse(num, num_target)
        net_loss += list(mse_loss.mean(dim=0).cpu().detach().numpy())
        mse_loss = mse_loss.mean()
        bce_loss = self.bce(bin, bin_target)

        net_loss += list(bce_loss.mean(dim=0).cpu().detach().numpy())
        bce_loss = bce_loss.mean()
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], codes[i])
            loss = loss.mean()
            cce_loss.append(loss)
            val = loss.cpu().item()
            net_loss += [val]
        if logging:
            if self.training:
                self.logger.training_step(net_loss)
            elif _id:
                self.logger.id_val_step(net_loss)
            elif not self.training:
                self.logger.val_step(net_loss)

        net_loss = np.array(net_loss).mean()
        return mse_loss, bce_loss, cce_loss, net_loss

    def do_backward(self, mse, bce, cce):

        mse.backward(retain_graph=True)
        bce.backward(retain_graph=True)
        for i, ls in enumerate(cce):
            if i == len(cce) - 1:
                ls.backward(retain_graph=False)
            else:
                ls.backward(retain_graph=True)

    def compute_baseline_performance(self, in_, out_):
        """
        Baseline performance is computed by generating a strong
            prediction for the identity function (predicting input==output)
            with a swapped (noisy) input,
            and computing the loss against the unaltered original data.

        This should be roughly the loss we expect when the encoder degenerates
            into the identity function solution.

        Returns net loss on baseline performance computation
            (sum of all losses)
        """
        self.eval()
        num_pred, bin_pred, codes = self.compute_targets(in_)
        bin_pred += ((bin_pred == 0).float() * 0.05)
        bin_pred -= ((bin_pred == 1).float() * 0.05)
        codes_pred = []
        for i, cd in enumerate(codes):
            feature = list(self.categorical_fts.items())[i][1]
            dim = len(feature['cats']) + 1
            pred = ohe(cd, dim, device=self.device) * 5
            codes_pred.append(pred)
        mse_loss, bce_loss, cce_loss, net_loss = self.compute_loss(num_pred, bin_pred, codes_pred, out_, logging=False)
        if isinstance(self.logger, BasicLogger):
            self.logger.baseline_loss = net_loss
        return net_loss

    def _create_stat_dict(self, a):
        scaler = self.loss_scaler()
        scaler.fit(a)
        return {'scaler': scaler}

    def fit(self, df, epochs=1, val=None, run_validation=False, use_val_for_loss_stats=False):
        """Does training.
        Args:
            df: pandas df used for training
            epochs: number of epochs to run training
            val: optional pandas dataframe for validation or loss stats
            run_validation: boolean indicating whether to collect validation loss for each
                epoch during training
            use_val_for_loss_stats: boolean indicating whether to use the validation set
                for loss statistics collection (for z score calculation)

        Raises:
            ValueError:
                if run_validation or use_val_for_loss_stats is True but val is not provided
        """
        if (run_validation or use_val_for_loss_stats) and val is None:
            raise ValueError("Validation set is required if either run_validation or \
                use_val_for_loss_stats is set to True.")

        if use_val_for_loss_stats:
            df_for_loss_stats = val.copy()
        else:
            # use train loss
            df_for_loss_stats = df.copy()

        if run_validation and val is not None:
            val = val.copy()

        if self.optim is None:
            self.build_model(df)
        if self.n_megabatches == 1:
            df = self.prepare_df(df)

        if run_validation and val is not None:
            val_df = self.prepare_df(val)
            val_in = val_df.swap(likelihood=self.swap_p)
            msg = "Validating during training.\n"
            msg += "Computing baseline performance..."
            baseline = self.compute_baseline_performance(val_in, val_df)
            if self.verbose:
                print(msg)
            result = []
            val_batches = len(val_df) // self.eval_batch_size
            if len(val_df) % self.eval_batch_size != 0:
                val_batches += 1

        n_updates = len(df) // self.batch_size
        if len(df) % self.batch_size > 0:
            n_updates += 1
        last_loss = 5000

        count_es = 0
        for i in range(epochs):
            self.train()
            if self.verbose:
                print(f'training epoch {i + 1}...')
            df = df.sample(frac=1.0)
            df = EncoderDataFrame(df)
            if self.n_megabatches > 1:
                self.train_megabatch_epoch(n_updates, df)
            else:
                input_df = df.swap(likelihood=self.swap_p)
                self.train_epoch(n_updates, input_df, df)

            if self.lr_decay is not None:
                self.lr_decay.step()

            if run_validation and val is not None:
                self.eval()
                with torch.no_grad():
                    swapped_loss = []
                    id_loss = []
                    for i in range(val_batches):
                        start = i * self.eval_batch_size
                        stop = (i + 1) * self.eval_batch_size

                        slc_in = val_in.iloc[start:stop]
                        slc_in_tensor = self.build_input_tensor(slc_in)

                        slc_out = val_df.iloc[start:stop]
                        slc_out_tensor = self.build_input_tensor(slc_out)

                        num, bin, cat = self.forward(slc_in_tensor)
                        _, _, _, net_loss = self.compute_loss(num, bin, cat, slc_out)
                        swapped_loss.append(net_loss)

                        num, bin, cat = self.forward(slc_out_tensor)
                        _, _, _, net_loss = self.compute_loss(num, bin, cat, slc_out, _id=True)
                        id_loss.append(net_loss)

                    # Early stopping
                    current_net_loss = net_loss
                    if self.verbose:
                        print('The Current Net Loss:', current_net_loss)

                    if current_net_loss > last_loss:
                        count_es += 1
                        if self.verbose:
                            print('Early stop count:', count_es)

                        if count_es >= self.patience:
                            if self.verbose:
                                print('Early stopping: early stop count({}) >= patience({})'.format(
                                    count_es, self.patience))
                            break

                    else:
                        if self.verbose:
                            print('Set count for earlystop: 0')
                        count_es = 0

                    last_loss = current_net_loss

                    self.logger.end_epoch()
                    #                     if self.project_embeddings:
                    #                         self.logger.show_embeddings(self.categorical_fts)
                    if self.verbose:
                        swapped_loss = np.array(swapped_loss).mean()
                        id_loss = np.array(id_loss).mean()

                        msg = '\n'
                        msg += 'net validation loss, swapped input: \n'
                        msg += f"{round(swapped_loss, 4)} \n\n"
                        msg += 'baseline validation loss: '
                        msg += f"{round(baseline, 4)} \n\n"
                        msg += 'net validation loss, unaltered input: \n'
                        msg += f"{round(id_loss, 4)} \n\n\n"
                        print(msg)

        #Getting training loss statistics
        # mse_loss, bce_loss, cce_loss, _ = self.get_anomaly_score(pdf) if pdf_val is None else self.get_anomaly_score(pd.concat([pdf, pdf_val]))
        mse_loss, bce_loss, cce_loss, _ = self.get_anomaly_score_with_losses(df_for_loss_stats)
        for i, ft in enumerate(self.numeric_fts):
            i_loss = mse_loss[:, i]
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)
        for i, ft in enumerate(self.binary_fts):
            i_loss = bce_loss[:, i]
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)
        for i, ft in enumerate(self.categorical_fts):
            i_loss = cce_loss[:, i]
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)

    def train_epoch(self, n_updates, input_df, df, pbar=None):
        """Run regular epoch."""

        if pbar is None and self.progress_bar:
            close = True
            pbar = tqdm.tqdm(total=n_updates)
        else:
            close = False

        for j in range(n_updates):

            start = j * self.batch_size
            stop = (j + 1) * self.batch_size
            in_sample = input_df.iloc[start:stop]
            in_sample_tensor = self.build_input_tensor(in_sample)
            target_sample = df.iloc[start:stop]
            num, bin, cat = self.forward(in_sample_tensor)
            mse, bce, cce, net_loss = self.compute_loss(num, bin, cat, target_sample, logging=True)
            self.do_backward(mse, bce, cce)
            self.optim.step()
            self.optim.zero_grad()

            if self.progress_bar:
                pbar.update(1)
        if close:
            pbar.close()

    def train_megabatch_epoch(self, n_updates, df):
        """
        Run epoch doing 'megabatch' updates, preprocessing data in large
        chunks.
        """
        if self.progress_bar:
            pbar = tqdm.tqdm(total=n_updates)
        else:
            pbar = None

        n_rows = len(df)
        n_megabatches = self.n_megabatches
        batch_size = self.batch_size
        res = n_rows / n_megabatches
        batches_per_megabatch = (res // batch_size) + 1
        megabatch_size = batches_per_megabatch * batch_size
        final_batch_size = n_rows - (n_megabatches - 1) * megabatch_size

        for i in range(n_megabatches):
            megabatch_start = int(i * megabatch_size)
            megabatch_stop = int((i + 1) * megabatch_size)
            megabatch = df.iloc[megabatch_start:megabatch_stop]
            megabatch = self.prepare_df(megabatch)
            input_df = megabatch.swap(self.swap_p)
            if i == (n_megabatches - 1):
                n_updates = int(final_batch_size // batch_size)
                if final_batch_size % batch_size > 0:
                    n_updates += 1
            else:
                n_updates = int(batches_per_megabatch)
            self.train_epoch(n_updates, input_df, megabatch, pbar=pbar)
            del megabatch
            del input_df
            gc.collect()

    def get_representation(self, df, layer=0):
        """
        Computes latent feature vector from hidden layer
            given input dataframe.

        argument layer (int) specifies which layer to get.
        by default (layer=0), returns the "encoding" layer.
            layer < 0 counts layers back from encoding layer.
            layer > 0 counts layers forward from encoding layer.
        """
        result = []
        n_batches = len(df) // self.eval_batch_size
        if len(df) % self.eval_batch_size != 0:
            n_batches += 1

        self.eval()
        if self.optim is None:
            self.build_model(df)
        df = self.prepare_df(df)
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.eval_batch_size
                stop = (i + 1) * self.eval_batch_size
                num, bin, embeddings = self.encode_input(df.iloc[start:stop])
                x = torch.cat(num + bin + embeddings, dim=1)
                if layer <= 0:
                    layers = len(self.encoder) + layer
                    x = self.encode(x, layers=layers)
                else:
                    x = self.encode(x)
                    x = self.decode(x, layers=layer)
                result.append(x)
        z = torch.cat(result, dim=0)
        return z

    def get_deep_stack_features(self, df):
        """
        records and outputs all internal representations
        of input df as row-wise vectors.
        Output is 2-d array with len() == len(df)
        """
        result = []

        n_batches = len(df) // self.eval_batch_size
        if len(df) % self.eval_batch_size != 0:
            n_batches += 1

        self.eval()
        if self.optim is None:
            self.build_model(df)
        df = self.prepare_df(df)
        with torch.no_grad():
            for i in range(n_batches):
                this_batch = []
                start = i * self.eval_batch_size
                stop = (i + 1) * self.eval_batch_size
                num, bin, embeddings = self.encode_input(df.iloc[start:stop])
                x = torch.cat(num + bin + embeddings, dim=1)
                for layer in self.encoder:
                    x = layer(x)
                    this_batch.append(x)
                for layer in self.decoder:
                    x = layer(x)
                    this_batch.append(x)
                z = torch.cat(this_batch, dim=1)
                result.append(z)
        result = torch.cat(result, dim=0)
        return result

    def get_anomaly_score(self, df):
        """
        Returns a per-row loss of the input dataframe.
        Does not corrupt inputs.
        """
        mse, bce, cce = self.get_anomaly_score_losses(df)

        combined_loss = torch.cat([mse, bce, cce], dim=1)

        net_loss = combined_loss.mean(dim=1).cpu().numpy()

        return net_loss

    def decode_to_df(self, x, df=None):
        """
        Runs input embeddings through decoder
        and converts outputs into a dataframe.
        """
        if df is None:
            cols = [x for x in self.binary_fts.keys()]
            cols += [x for x in self.numeric_fts.keys()]
            cols += [x for x in self.categorical_fts.keys()]
            df = pd.DataFrame(index=range(len(x)), columns=cols)

        num, bin, cat = self.decode(x)

        num_cols = [x for x in self.numeric_fts.keys()]
        num_df = pd.DataFrame(data=num.cpu().numpy(), index=df.index)
        num_df.columns = num_cols
        for ft in num_df.columns:
            feature = self.numeric_fts[ft]
            col = num_df[ft]
            trans_col = feature['scaler'].inverse_transform(col.values)
            result = pd.Series(index=df.index, data=trans_col)
            num_df[ft] = result

        bin_cols = [x for x in self.binary_fts.keys()]
        bin_df = pd.DataFrame(data=bin.cpu().numpy(), index=df.index)
        bin_df.columns = bin_cols
        bin_df = bin_df.apply(lambda x: round(x)).astype(bool)
        for ft in bin_df.columns:
            feature = self.binary_fts[ft]
            map = {False: feature['cats'][0], True: feature['cats'][1]}
            bin_df[ft] = bin_df[ft].apply(lambda x: map[x])

        cat_df = pd.DataFrame(index=df.index)
        for i, ft in enumerate(self.categorical_fts):
            feature = self.categorical_fts[ft]
            cats = feature['cats']

            if (len(cats) > 0):
                # get argmax excluding NaN column (impute with next-best guess)
                codes = torch.argmax(cat[i][:, :-1], dim=1).cpu().numpy()
            else:
                # Only one option
                codes = torch.argmax(cat[i], dim=1).cpu().numpy()
            cat_df[ft] = codes
            cats = feature['cats'] + ["_other"]
            cat_df[ft] = cat_df[ft].apply(lambda x: cats[x])

        # concat
        output_df = pd.concat([num_df, bin_df, cat_df], axis=1)

        return output_df[df.columns]

    def df_predict(self, df):
        """
        Runs end-to-end model.
        Interprets output and creates a dataframe.
        Outputs dataframe with same shape as input
            containing model predictions.
        """
        self.eval()
        data = self.prepare_df(df)
        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            x = self.encode(x)
            output_df = self.decode_to_df(x, df=df)

        return output_df

    def get_anomaly_score_with_losses(self, df):

        mse, bce, cce = self.get_anomaly_score_losses(df)

        net = self.get_anomaly_score(df)

        return mse, bce, cce, net

    def get_anomaly_score_losses(self, df):
        """
        Run the input dataframe `df` through the autoencoder to get the recovery losses by feature type
        (numerical/boolean/categorical).
        """
        self.eval()

        n_batches = len(df) // self.batch_size
        if len(df) % self.batch_size > 0:
            n_batches += 1

        mse_loss_slices, bce_loss_slices, cce_loss_slices = [], [], []
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.batch_size
                stop = (i + 1) * self.batch_size

                df_slice = df.iloc[start:stop]
                data_slice = self.prepare_df(df_slice)
                num_target, bin_target, codes = self.compute_targets(data_slice)

                input_slice = self.build_input_tensor(data_slice)

                num, bin, cat = self.forward(input_slice)
                mse_loss_slice: torch.Tensor = self.mse(num, num_target)
                bce_loss_slice: torch.Tensor = self.bce(bin, bin_target)
                cce_loss_slice_of_each_feat = [
                ]  # each entry in this list is the cce loss of a feature, ordered by the feature list self.categorical_fts
                for i, ft in enumerate(self.categorical_fts):
                    loss = self.cce(cat[i], codes[i])
                    # Convert to 2 dimensions
                    cce_loss_slice_of_each_feat.append(loss.data.reshape(-1, 1))
                cce_loss_slice = torch.cat(cce_loss_slice_of_each_feat,
                                           dim=1)  # merge the tensors into one (n_records * n_features) tensor

                mse_loss_slices.append(mse_loss_slice)
                bce_loss_slices.append(bce_loss_slice)
                cce_loss_slices.append(cce_loss_slice)

        mse_loss = torch.cat(mse_loss_slices, dim=0)
        bce_loss = torch.cat(bce_loss_slices, dim=0)
        cce_loss = torch.cat(cce_loss_slices, dim=0)
        return mse_loss, bce_loss, cce_loss

    def scale_losses(self, mse, bce, cce):

        # Create outputs
        mse_scaled = torch.zeros_like(mse)
        bce_scaled = torch.zeros_like(bce)
        cce_scaled = torch.zeros_like(cce)

        for i, ft in enumerate(self.numeric_fts):
            mse_scaled[:, i] = self.feature_loss_stats[ft]['scaler'].transform(mse[:, i])

        for i, ft in enumerate(self.binary_fts):
            bce_scaled[:, i] = self.feature_loss_stats[ft]['scaler'].transform(bce[:, i])

        for i, ft in enumerate(self.categorical_fts):
            cce_scaled[:, i] = self.feature_loss_stats[ft]['scaler'].transform(cce[:, i])

        return mse_scaled, bce_scaled, cce_scaled

    def get_results(self, df, return_abs=False):
        pdf = pd.DataFrame()
        self.eval()

        data = self.prepare_df(df)

        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            x = self.encode(x)
            output_df = self.decode_to_df(x)

        # set the index of the prediction df to match the input df
        output_df.index = df.index

        mse, bce, cce = self.get_anomaly_score_losses(df)
        mse_scaled, bce_scaled, cce_scaled = self.scale_losses(mse, bce, cce)

        if (return_abs):
            mse_scaled = abs(mse_scaled)
            bce_scaled = abs(bce_scaled)
            cce_scaled = abs(cce_scaled)

        combined_loss = torch.cat([mse_scaled, bce_scaled, cce_scaled], dim=1)

        for i, ft in enumerate(self.numeric_fts):
            pdf[ft] = df[ft]
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = mse[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = mse_scaled[:, i].cpu().numpy()

        for i, ft in enumerate(self.binary_fts):
            pdf[ft] = df[ft]
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = bce[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = bce_scaled[:, i].cpu().numpy()

        for i, ft in enumerate(self.categorical_fts):
            pdf[ft] = df[ft]
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = cce[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = cce_scaled[:, i].cpu().numpy()

        pdf['max_abs_z'] = combined_loss.max(dim=1)[0].cpu().numpy()
        pdf['mean_abs_z'] = combined_loss.mean(dim=1).cpu().numpy()

        # add a column describing the scaler of the losses
        if self.loss_scaler_str == 'standard':
            output_scaled_loss_str = 'z'
        elif self.loss_scaler_str == 'modified':
            output_scaled_loss_str = 'modz'
        else:
            # in case other custom scaling is used
            output_scaled_loss_str = f'{self.loss_scaler_str}_scaled'
        pdf['z_loss_scaler_type'] = output_scaled_loss_str

        return pdf
