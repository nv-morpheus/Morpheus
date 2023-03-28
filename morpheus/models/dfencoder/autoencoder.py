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
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import tqdm

from .ae_module import AEModule
from .dataframe import EncoderDataFrame
from .distributed_ae import DistributedAutoEncoder
from .logging import BasicLogger
from .logging import IpynbLogger
from .logging import TensorboardXLogger
from .scalers import GaussRankScaler
from .scalers import ModifiedScaler
from .scalers import NullScaler
from .scalers import StandardScaler


def ohe(input_vector, dim, device="cpu"):
    """Does one-hot encoding of input vector."""
    batch_size = len(input_vector)
    nb_digits = dim

    y = input_vector.reshape(-1, 1)
    y_onehot = torch.FloatTensor(batch_size, nb_digits).to(device)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot


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
            distributed_training=False,
            logger='basic',
            logdir='logdir/',
            project_embeddings=True,
            run=None,
            progress_bar=True,
            n_megabatches=1,
            scaler='standard',
            patience=5,
            preset_cats=None,
            preset_numerical_scaler_params=None,
            binary_feature_list=None,
            loss_scaler='standard',  # scaler for the losses (z score)
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.numeric_fts = OrderedDict()
        self.binary_fts = OrderedDict()
        self.categorical_fts = OrderedDict()
        self.cyclical_fts = OrderedDict()
        self.feature_loss_stats = dict()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.distributed_training = distributed_training

        self.model = AEModule(
            verbose=verbose,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            encoder_activations=encoder_activations,
            decoder_activations=decoder_activations,
            activation=activation,
            device=self.device,
            *args,
            **kwargs,
        )
        self.optimizer = optimizer
        self.optim = None
        self.lr = lr
        self.lr_decay = lr_decay

        self.min_cats = min_cats
        self.preset_cats = preset_cats
        self.preset_numerical_scaler_params = preset_numerical_scaler_params

        self.swap_p = swap_p
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.numeric_output = None
        self.binary_output = None

        # `num_names` is a list of column names that contain numeric data (int & float fields).
        self.num_names = []
        self.bin_names = binary_feature_list

        self.amsgrad = amsgrad
        self.momentum = momentum
        self.betas = betas
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self.progress_bar = progress_bar

        self.mse = torch.nn.modules.loss.MSELoss(reduction='none')
        self.bce = torch.nn.modules.loss.BCELoss(reduction='none')
        self.cce = torch.nn.modules.loss.CrossEntropyLoss(reduction='none')

        self.verbose = verbose

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

    def _init_numeric(self, df=None):
        """
        Initializes the numerical features of the model by either using preset numerical scaler parameters
        or by using the input data.

        Args:
        df (pandas DataFrame, optional): The input data to be used to initialize the numerical features.
        If not provided, self.preset_numerical_scaler_params must be provided.

        Raises:
        ValueError: If both df and self.preset_numerical_scaler_params are not provided.
        """
        if df is None and self.preset_numerical_scaler_params is None:
            raise ValueError("Either `df` or `self.preset_numerical_scaler_params` needs to be provided.")

        if self.preset_numerical_scaler_params:
            if self.verbose:
                print("Using self.preset_numerical_scaler_params to override the numerical scalers...")
            for ft, scaler_params in self.preset_numerical_scaler_params.items():
                # scaler_params should include the following keys: scaler_type, scaler_attr_dict, mean, std
                scaler = self.get_scaler(scaler_params.get("scaler_type", "gauss_rank"))()
                for k, v in scaler_params["scaler_attr_dict"].items():
                    # scaler_params['scaler_attr_dict'] should be a dict including all the class attributes of a fitted scaler class
                    setattr(scaler, k, v)
                feature = {
                    "mean": scaler_params["mean"],
                    "std": scaler_params["std"],
                    "scaler": scaler,
                }
                self.numeric_fts[ft] = feature
        else:
            # initialize using a dataframe
            dt = df.dtypes
            numeric = []
            numeric += list(dt[dt == int].index)
            numeric += list(dt[dt == float].index)

            if isinstance(self.scaler, str):
                scalers = {ft: self.scaler for ft in numeric}
            elif isinstance(self.scaler, dict):
                scalers = self.scaler

            for ft in numeric:
                Scaler = self.get_scaler(scalers.get(ft, "gauss_rank"))
                feature = {
                    "mean": df[ft].mean(),
                    "std": df[ft].std(),
                    "scaler": Scaler(),
                }
                feature["scaler"].fit(df[ft][~df[ft].isna()].values)
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

    def _init_cats(self, df):
        dt = df.dtypes
        objects = list(dt[dt == "object"].index)
        for ft in objects:
            feature = {}
            vl = df[ft].value_counts()
            cats = list(vl[vl >= self.min_cats].index)
            feature['cats'] = cats
            self.categorical_fts[ft] = feature

    def _init_binary(self, df=None):
        """
        Initializes the binary features of the model.

        Args:
        df (pandas DataFrame, optional): The input data to be used to initialize the binary features.
        If not provided, self.bin_names must be provided.

        Raises:
        ValueError: If both df and self.bin_names are not provided.
        """
        if df is None and self.bin_names is None:
            raise ValueError("Need to provide one of the two params (df or binary_features). \
                If there is no binary feartures, try providing the parameter `binary_feature_list=[]` during class init."
                             )

        if self.bin_names is not None:
            if self.verbose:
                print("Using the preset binary feature list `self.bin_names` to initialize the binary features...")
            binaries = self.bin_names
        else:
            dt = df.dtypes
            binaries = list(dt[dt == bool].index)
            self.bin_names = binaries

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

    def _init_features(self, df=None):
        if self.preset_cats is not None:
            if self.verbose:
                print('Using the preset categories `self.preset_cats` to initialize the categories features...')
            self.categorical_fts = self.preset_cats
        else:
            self._init_cats(df)
        self._init_numeric(df)
        self._init_binary(df)

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

    def _build_model(self, df=None, rank=None):
        """
        Builds the autoencoder model using either the given dataframe or the preset feature information for metadata.
        If distributed training is enabled (self.distributed_training is True), wraps the pytorch module with DDP.
        User should not need to call this function directly as it's called before training in the fit() functions.

        Args:
            df (dataframe, optional): the input dataframe to be used to infer metadata
            rank (int, optional): rank of the process being used for distributed training,
                used only if distributed_training=True
        """
        if self.verbose:
            print('Building model...')

        # get metadata from features
        self._init_features(df)

        self.model.build(self.numeric_fts, self.binary_fts, self.categorical_fts)
        if self.distributed_training:
            if rank is None:
                raise ValueError('`rank` missing. `rank` is required for distributed training.')

            self.model._ddp_params_and_buffers_to_ignore = []
            if len(self.numeric_fts) == 0:
                # if there is no numeric feature, ignore this layer to avoid errors while syncing parameters across gpus
                self.model._ddp_params_and_buffers_to_ignore.append('numeric_output.weight')
            if len(self.binary_fts) == 0:
                # if there is no binary feature, ignore this layer to avoid errors while syncing parameters across gpus
                self.model._ddp_params_and_buffers_to_ignore.append('binary_output.weight')

            self.model = DistributedAutoEncoder(self.model, device_ids=[rank], output_device=rank)

        self._build_optimizer()
        if self.lr_decay is not None:
            self.lr_decay = torch.optim.lr_scheduler.ExponentialLR(self.optim, self.lr_decay)

        self._build_logger()

        if self.verbose:
            print('done!')

    def _build_optimizer(self):
        lr = self.lr
        params = self.model.parameters()
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(params,
                                     lr=self.lr,
                                     amsgrad=self.amsgrad,
                                     weight_decay=self.weight_decay,
                                     betas=self.betas)
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                params,
                lr,
                momentum=self.momentum,
                nesterov=self.nesterov,
                dampening=self.dampening,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError('Provided optimizer unsupported. Supported optimizers include: [adam, sgd].')
        self.optim = optim

    def _build_logger(self):
        """ Initializes the logger to be used for training the model."""
        cat_names = list(self.categorical_fts.keys())
        fts = self.num_names + self.bin_names + cat_names
        if self.logger == 'basic':
            self.logger = BasicLogger(fts=fts)
        elif self.logger == 'ipynb':
            self.logger = IpynbLogger(fts=fts)
        elif self.logger == 'tensorboard':
            self.logger = TensorboardXLogger(logdir=self.logdir, run=self.run, fts=fts)

    def compute_targets(self, df):
        num = torch.tensor(df[self.num_names].values).float().to(self.device)
        bin = torch.tensor(df[self.bin_names].astype(int).values).float().to(self.device)
        codes = []
        for ft in self.categorical_fts:
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
        for i, embedding_layer in enumerate(self.model.categorical_embedding.values()):
            emb = embedding_layer(codes[i])
            embeddings.append(emb)
        return [num], [bin], embeddings

    def build_input_tensor(self, df):
        num, bin, embeddings = self.encode_input(df)
        x = torch.cat(num + bin + embeddings, dim=1)
        return x

    def preprocess_train_data(self, df, shuffle_rows_in_batch=True):
        """ Wrapper function round `self.preprocess_data` feeding in the args suitable for a training set."""
        return self.preprocess_data(
            df,
            shuffle_rows_in_batch=shuffle_rows_in_batch,
            include_original_input_tensor=False,
            include_swapped_input_by_feature_type=False,
        )

    def preprocess_validation_data(self, df, shuffle_rows_in_batch=False):
        """ Wrapper function round `self.preprocess_data` feeding in the args suitable for a validation set."""
        return self.preprocess_data(
            df,
            shuffle_rows_in_batch=shuffle_rows_in_batch,
            include_original_input_tensor=True,
            include_swapped_input_by_feature_type=True,
        )

    def preprocess_data(
        self,
        df,
        shuffle_rows_in_batch,
        include_original_input_tensor,
        include_swapped_input_by_feature_type,
    ):
        """
        Preprocesses a pandas dataframe `df` for input into the autoencoder model.

        Args:
            df (pandas dataframe): the input dataframe to preprocess
            shuffle_rows_in_batch (bool): whether to shuffle the rows of the dataframe before processing
            include_original_input_tensor (bool): whether to process the df into an input tensor without swapping and include
                it in the returned data dict.
                Note. Training required only the swapped input tensor while validation can use both.
            include_swapped_input_by_feature_type (bool): whether to process the swapped df into num/bin/cat feature tensors and include them
            in the returned data dict. This is useful for baseline performance evaluation for validation.

        Returns:
            preprocessed_data (dict): a dict containing the preprocessed input data and targets by feature type
        """
        df = self.prepare_df(df)
        if shuffle_rows_in_batch:
            df = df.sample(frac=1.0)
        df = EncoderDataFrame(df)
        swapped_df = df.swap(likelihood=self.swap_p)
        swapped_input_tensor = self.build_input_tensor(swapped_df)
        num_target, bin_target, codes = self.compute_targets(df)

        preprocessed_data = {
            'input_swapped': swapped_input_tensor,
            'num_target': num_target,
            'bin_target': bin_target,
            'cat_target': codes,
            'size': len(df),
        }

        if include_original_input_tensor:
            preprocessed_data['input_original'] = self.build_input_tensor(df)

        if include_swapped_input_by_feature_type:
            num_swapped, bin_swapped, codes_swapped = self.compute_targets(swapped_df)
            preprocessed_data['num_swapped'] = num_swapped
            preprocessed_data['bin_swapped'] = bin_swapped
            preprocessed_data['cat_swapped'] = codes_swapped

        return preprocessed_data

    def compute_loss(self, num, bin, cat, target_df, logging=True, _id=False):
        num_target, bin_target, codes = self.compute_targets(target_df)
        return self.compute_loss_from_targets(
            num=num,
            bin=bin,
            cat=cat,
            num_target=num_target,
            bin_target=bin_target,
            cat_target=codes,
            logging=logging,
            _id=_id,
        )

    def compute_loss_from_targets(self, num, bin, cat, num_target, bin_target, cat_target, logging=True, _id=False):
        """
        Computes the loss from targets.

        Args:
            num (tensor): numerical data tensor
            bin (tensor): binary data tensor
            cat (list of tensors): list of categorical data tensors
            num_target (tensor): target numerical data tensor
            bin_target (tensor): target binary data tensor
            cat_target (list of tensors): list of target categorical data tensors
            logging (bool): whether to log the loss in self.logger
            _id (bool): whether the current step is an id validation step (for logging)

        Returns:
            tuple: A tuple containing the mean mse/bce losses, list of mean cce losses, and mean net loss
        """
        if logging:
            if self.logger is not None:
                logging = True
            else:
                logging = False
        net_loss = []
        mse_loss = self.mse(num, num_target)
        net_loss += list(mse_loss.mean(dim=0).cpu().detach().numpy())
        mse_loss = mse_loss.mean()
        bce_loss = self.bce(bin, bin_target)

        net_loss += list(bce_loss.mean(dim=0).cpu().detach().numpy())
        bce_loss = bce_loss.mean()
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], cat_target[i])
            loss = loss.mean()
            cce_loss.append(loss)
            val = loss.cpu().item()
            net_loss += [val]
        if logging:
            if self.model.training:
                self.logger.training_step(net_loss)
            elif _id:
                self.logger.id_val_step(net_loss)
            elif not self.model.training:
                self.logger.val_step(net_loss)

        net_loss = np.array(net_loss).mean()
        return mse_loss, bce_loss, cce_loss, net_loss

    def do_backward(self, mse, bce, cce):
        # running `backward()` seperately on mse/bce/cce is equivalent to summing them up and run `backward()` once
        loss_fn = mse + bce
        for ls in cce:
            loss_fn += ls
        loss_fn.backward()

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
        self.model.eval()

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

    def fit(
        self,
        train_data,
        epochs=1,
        val_data=None,
        run_validation=False,
        use_val_for_loss_stats=False,
        rank=None,
        world_size=None,
    ):
        """
        Does training in the specified mode (indicated by self.distrivuted_training).

        Args:
            train_data: pandas df (centralized) or pytorch dataloader (distributed) used
                for training
            epochs: number of epochs to run training
            val_data: optional pandas dataframe (centralized) or pytorch dataset (distributed)
                for validation and loss stats
            run_validation: boolean indicating whether to collect validation loss for each
                epoch during training
            use_val_for_loss_stats: boolean indicating whether to use the validation set
                for loss statistics collection (for z score calculation)
            rank: int, the rank of the current process
            world_size: int, the total number of processes
        Raises:
            ValueError:
                if run_validation or use_val_for_loss_stats is True but val is not provided

        """
        if not self.distributed_training:
            self._fit_centralized(
                df=train_data,
                epochs=epochs,
                val=val_data,
                run_validation=run_validation,
                use_val_for_loss_stats=use_val_for_loss_stats,
            )
        else:
            # distributed training requires rank and world_size
            if rank is None or world_size is None:
                raise ValueError('`rank` and `world_size` must be provided for distributed training.')
            self._fit_distributed(
                train_dataloader=train_data,
                epochs=epochs,
                val_dataset=val_data,
                run_validation=run_validation,
                use_val_for_loss_stats=use_val_for_loss_stats,
                rank=rank,
                world_size=world_size,
            )

    def _fit_centralized(self, df, epochs=1, val=None, run_validation=False, use_val_for_loss_stats=False):
        """Does training in a single process on a single GPU.
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
            self._build_model(df)

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

            val_batches = len(val_df) // self.eval_batch_size
            if len(val_df) % self.eval_batch_size != 0:
                val_batches += 1

        n_updates = len(df) // self.batch_size
        if len(df) % self.batch_size > 0:
            n_updates += 1
        last_loss = 5000

        count_es = 0
        for i in range(epochs):
            self.model.train()

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
                self.model.eval()
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

                        num, bin, cat = self.model(slc_in_tensor)
                        _, _, _, net_loss = self.compute_loss(num, bin, cat, slc_out)
                        swapped_loss.append(net_loss)

                        num, bin, cat = self.model(slc_out_tensor)
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

    def _fit_distributed(
        self,
        train_dataloader,
        rank,
        world_size,
        epochs=1,
        val_dataset=None,
        run_validation=False,
        use_val_for_loss_stats=True,
    ):
        """
        Fit the model in the distributed fashion with early stopping based on validation loss.
        If run_validation is True, the val_dataset will be used for validation during training
        and early stopping will be applied based on patience argument.

        Args:
            train_dataloader (pytorch dataloader): dataloader object of training data
            rank (int): the rank of the current process
            world_size (int): the total number of processes
            epochs (int, optional): the number of epochs to train for
            val_dataset (pytorch dataset or dataloader, optional): the validation dataset
                (with __iter__() that yields a batch at a time)
            run_validation (bool, optional): whether to perform validation during training
            use_val_for_loss_stats (bool, optional): whether to populate loss stats in the
                main process (rank 0) for z-score calculation using the validation set.
                If set to False, loss stats would be populated using the train_dataloader,
                which can be slow due to data size.
                `True` is the default as using the validation set to populate loss stats is
                strongly recommended (for both efficiency and model efficacy).
        """
        if run_validation and val_dataset is None:
            raise ValueError("`run_validation` is set to True but the validation set (val_dataset) is not provided.")

        if use_val_for_loss_stats and val_dataset is None:
            raise ValueError("Validation set is required if either run_validation or \
                use_val_for_loss_stats is set to True.")

        if self.optim is None:
            self._build_model(rank=rank)

        is_main_process = rank == 0
        should_run_validation = (run_validation and val_dataset is not None)
        if self.patience and not should_run_validation:
            print(
                f'WARNING: Not going to perform early-stopping. self.patience(={self.patience}) is provided for early-stopping\
                 but validation is not enabled. Please set `run_validation` to True and provide a `val_dataset` to enable early-stopping.'
            )

        if is_main_process and should_run_validation:
            if self.verbose:
                print('Validating during training. Computing baseline performance...')
            baseline = self._compute_baseline_performance_from_dataset(val_dataset)

            if isinstance(self.logger, BasicLogger):
                self.logger.baseline_loss = baseline

            if self.verbose:
                print(f'Baseline loss: {round(baseline, 4)}')

        # early stopping
        count_es = 0
        last_val_loss = float('inf')
        should_early_stop = False
        for epoch in range(epochs):
            if self.verbose:
                print(f'Rank{rank} training epoch {epoch + 1}...')

            # if we are using DistributedSampler, we have to tell it which epoch this is
            train_dataloader.sampler.set_epoch(epoch)

            train_loss_sum = 0
            train_loss_count = 0
            for data_d in train_dataloader:
                loss = self._fit_batch(**data_d['data'])

                train_loss_count += 1
                train_loss_sum += loss

            if self.lr_decay is not None:
                self.lr_decay.step()

            if is_main_process and should_run_validation:
                # run validation
                curr_val_loss = self._validate_dataset(val_dataset, rank)
                if self.verbose:
                    print(f'Rank{rank} Loss: {round(last_val_loss, 4)}->{round(curr_val_loss, 4)}')

                if self.patience:  # early stopping
                    if curr_val_loss > last_val_loss:
                        count_es += 1
                        if self.verbose:
                            print(f'Rank{rank} Loss went up. Early stop count: {count_es}')

                        if count_es >= self.patience:
                            if self.verbose:
                                print(f'Early stopping: early stop count({count_es}) >= patience({self.patience})')
                            should_early_stop = True
                    else:
                        if self.verbose:
                            print(f'Rank{rank} Loss went down. Reset count for earlystop to 0')
                        count_es = 0

                    last_val_loss = curr_val_loss

                self.logger.end_epoch()

            # sync early stopping info so the early stopping decision can be passed from the main process to other processes
            early_stpping_state = [None for _ in range(world_size)
                                   ]  # we have to create enough room to store the collected objects
            torch.distributed.all_gather_object(early_stpping_state, should_early_stop)
            should_early_stop_synced = early_stpping_state[0]  # take the state of the main process
            if should_early_stop_synced is True:
                if self.verbose:
                    print(f'Rank{rank} Early stopped.')
                break

        if is_main_process:
            dataset_for_loss_stats = val_dataset if use_val_for_loss_stats else train_dataloader
            self._populate_loss_stats_from_dataset(dataset_for_loss_stats)

    def _fit_batch(self, input_swapped, num_target, bin_target, cat_target, **kwargs):
        """
        Forward pass on the input_swapped, then computes the losses from the predicted outputs and actual targets,
        performs backpropagation, updates the model parameters, and returns the net loss.

        Args:
            input_swapped (tensor): input tensor of shape (batch_size, feature vector size), some values are randomly swapped for denoising
            num_target (tensor): tensor of shape (batch_size, numerical feature count) with numerical targets
            bin_target (tensor): tensor of shape (batch_size, binary feature count) with binary targets
            cat_target (list): list of size (categorical feature count), each entry is a 1-d tensor of shape (batch_size) containing the categorical targets

        Returns:
            net_loss (float): total loss computed as the weighted sum of the mse, bce and cce losses
        """
        self.model.train()
        num, bin, cat = self.model(input_swapped)
        mse, bce, cce, net_loss = self.compute_loss_from_targets(
            num=num,
            bin=bin,
            cat=cat,
            num_target=num_target,
            bin_target=bin_target,
            cat_target=cat_target,
            logging=True,
        )
        self.do_backward(mse, bce, cce)
        self.optim.step()
        self.optim.zero_grad()
        return net_loss

    def _compute_baseline_performance_from_dataset(self, val_dataset):
        self.model.eval()
        loss_sum = 0
        sample_count = 0
        with torch.no_grad():
            for data_d in val_dataset:
                curr_batch_size = data_d['data']['size']
                loss = self._compute_batch_baseline_performance(**data_d['data'])
                loss_sum += loss
                sample_count += curr_batch_size

        baseline = loss_sum / sample_count
        return baseline

    def _compute_batch_baseline_performance(
            self,
            num_swapped,
            bin_swapped,
            cat_swapped,
            num_target,
            bin_target,
            cat_target,
            **kwargs,  # ignore other unused kwargs
    ):
        bin_swapped += ((bin_swapped == 0).float() * 0.05)
        bin_swapped -= ((bin_swapped == 1).float() * 0.05)
        codes_swapped_ohe = []
        for cd, feature in zip(cat_swapped, self.categorical_fts.values()):
            dim = len(feature['cats']) + 1
            cd_ohe = ohe(cd, dim, device=self.device) * 5
            codes_swapped_ohe.append(cd_ohe)

        _, _, _, net_loss = self.compute_loss_from_targets(
            num=num_swapped,
            bin=bin_swapped,
            cat=codes_swapped_ohe,
            num_target=num_target,
            bin_target=bin_target,
            cat_target=cat_target,
            logging=False
        )
        return net_loss

    def _validate_dataset(self, val_dataset, rank=None):
        """
        Runs a validation loop on the given validation dataset, computing and returning the average loss of both the original
        input and the input with swapped values.

        Args:
            val_dataset (torch.utils.data.Dataset): validation dataset to be used for validation
            rank (int): optional rank of the process being used for distributed training, used only for logging

        Returns:
            float: the average loss of the original input in the validation dataset
        """
        self.model.eval()
        with torch.no_grad():
            swapped_loss = []
            id_loss = []
            for data_d in val_dataset:
                orig_net_loss, net_loss = self._validate_batch(**data_d['data'])
                id_loss.append(orig_net_loss)
                swapped_loss.append(net_loss)

            swapped_loss = np.array(swapped_loss).mean()
            id_loss = np.array(id_loss).mean()

            if self.verbose:
                rank_str = '' if rank is None else f'R{rank} '
                print(f'\t{rank_str}Swapped loss: {round(swapped_loss, 4)}, Orig. loss: {round(id_loss, 4)}')
        return id_loss

    def _validate_batch(self, input_original, input_swapped, num_target, bin_target, cat_target, **kwargs):
        """
        Forward pass on the validation inputs, then computes the losses from the predicted outputs and actual targets,
        and returns the net loss.

        Args:
            input_original (tensor): input tensor of shape (batch_size, feature vector size)
            input_swapped (tensor): input tensor of shape (batch_size, feature vector size), some values are randomly swapped for denoising
            num_target (tensor): tensor of shape (batch_size, numerical feature count) with numerical targets
            bin_target (tensor): tensor of shape (batch_size, binary feature count) with binary targets
            cat_target (list): list of size (categorical feature count), each entry is a 1-d tensor of shape (batch_size) containing the categorical targets

        Returns:
            A tuple containing two floats:
            - orig_net_loss: the net loss when passing `input_original` through the model
            - net_loss: the net loss when passing the `input_swapped` through the model
        """
        orig_num, orig_bin, orig_cat = self.model(input_original)
        _, _, _, orig_net_loss = self.compute_loss_from_targets(
            num=orig_num,
            bin=orig_bin,
            cat=orig_cat,
            num_target=num_target,
            bin_target=bin_target,
            cat_target=cat_target,
            logging=True,
        )

        num, bin, cat = self.model(input_swapped)
        _, _, _, net_loss = self.compute_loss_from_targets(
            num=num,
            bin=bin,
            cat=cat,
            num_target=num_target,
            bin_target=bin_target,
            cat_target=cat_target,
            logging=True,
        )
        return orig_net_loss, net_loss

    def _populate_loss_stats_from_dataset(self, dataset):
        """
        Populates the `self.feature_loss_stats` dict with feature losses computed using the provided dataset.
        Args:
            dataset (torch.utils.data.Dataset): dataset to compute the feature losses for
        """
        self.model.eval()
        feature_losses = self._get_feature_losses_from_dataset(dataset)
        # populate loss stats
        for ft, losses in feature_losses.items():
            loss = losses.cpu().numpy()
            self.feature_loss_stats[ft] = self._create_stat_dict(loss)

    def _get_feature_losses_from_dataset(self, dataset):
        """
        Computes the feature losses for each feature in the model for a given dataset.
        Args:
            dataset (torch.utils.data.Dataset): dataset to compute the feature losses for

        Returns:
            feature_losses (dict): a dict mapping feature names to a tensor of losses
        """
        feature_losses = defaultdict(list)
        with torch.no_grad():
            for data_d in dataset:
                batch_feature_losses = self._get_batch_feature_losses(**data_d['data'])
                for ft, loss_l in batch_feature_losses.items():
                    feature_losses[ft].append(loss_l)
        return {ft: torch.cat(tensor_l, dim=0) for ft, tensor_l in feature_losses.items()}

    def _get_batch_feature_losses(self, input_original, num_target, bin_target, cat_target, **kwargs):
        """
        Calculates the feature-wise losses for a batch of input data.

        Args:
            input_original (tensor): input tensor of shape (batch_size, feature vector size)
            num_target (tensor): tensor of shape (batch_size, numerical feature count) with numerical targets
            bin_target (tensor): tensor of shape (batch_size, binary feature count) with binary targets
            cat_target (list): list of size (categorical feature count), each entry is a 1-d tensor of shape (batch_size) containing the categorical targets

        Returns:
            batch_feature_losses (dict): a dict mapping feature names to a tensor of losses for the batch
        """
        batch_feature_losses = {}

        num, bin, cat = self.model(input_original)
        mse_loss = self.mse(num, num_target)
        for i, ft in enumerate(self.numeric_fts):
            batch_feature_losses[ft] = mse_loss[:, i]

        bce_loss = self.bce(bin, bin_target)
        for i, ft in enumerate(self.binary_fts):
            batch_feature_losses[ft] = bce_loss[:, i]

        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], cat_target[i])
            batch_feature_losses[ft] = loss

        return batch_feature_losses

    def get_results_from_dataset(self, dataset, preloaded_df, return_abs=False):
        """
        Returns a pandas dataframe of inference results and losses for a given dataset.
        Note. this function requires the whole inference set to be in loaded into memory as a pandas df

        Args:
            dataset (torch.utils.data.Dataset): dataset for inference
            preloaded_df: a pandas dataframe that contains the original data
            return_abs: a boolean flag indicating whether the absolute value of the loss scalers should be returned.

        Returns:
            result (pandas dataframe): inference result with losses of each feature
        """
        result = pd.DataFrame()

        if self.verbose:
            print(f'Getting inference results... (total of {len(dataset)} batches)')

        self.model.eval()
        feature_losses = defaultdict(list)
        output_df = []
        with torch.no_grad():
            for step, data_d in enumerate(dataset):
                if self.verbose:
                    print(f'\tinferencing batch {step}...')

                batch_feature_losses = self._get_batch_feature_losses(**data_d['data'])
                for ft, loss_l in batch_feature_losses.items():
                    feature_losses[ft].append(loss_l)

                num, bin, cat = self.model(data_d['data']['input_original'])
                batch_output_df = self.decode_outputs_to_df(num=num, bin=bin, cat=cat)
                output_df.append(batch_output_df)

        if self.verbose:
            print(f'\tDone running inference. Making output df...')

        feature_losses = {ft: torch.cat(tensor_l, dim=0) for ft, tensor_l in feature_losses.items()}
        output_df = pd.concat(output_df).reset_index(drop=True)

        for ft, loss_tensor in feature_losses.items():
            result[ft] = preloaded_df[ft]
            result[ft + '_pred'] = output_df[ft]
            result[ft + '_loss'] = loss_tensor.cpu().numpy()
            z_loss = self.feature_loss_stats[ft]['scaler'].transform(loss_tensor)
            if return_abs:
                z_loss = abs(z_loss)
            result[ft + '_z_loss'] = z_loss.cpu().numpy()

        result['max_abs_z'] = result[[f'{ft}_z_loss' for ft in feature_losses]].max(axis=1)
        result['mean_abs_z'] = result[[f'{ft}_z_loss' for ft in feature_losses]].mean(axis=1)

        # add a column describing the scaler of the losses
        if self.loss_scaler_str == 'standard':
            output_scaled_loss_str = 'z'
        elif self.loss_scaler_str == 'modified':
            output_scaled_loss_str = 'modz'
        else:
            # in case other custom scaling is used
            output_scaled_loss_str = f'{self.loss_scaler_str}_scaled'
        result['z_loss_scaler_type'] = output_scaled_loss_str

        return result

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
            num, bin, cat = self.model(in_sample_tensor)  # forward
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

        self.model.eval()

        if self.optim is None:
            self._build_model(df)
        df = self.prepare_df(df)
        with torch.no_grad():
            for i in range(n_batches):
                start = i * self.eval_batch_size
                stop = (i + 1) * self.eval_batch_size
                num, bin, embeddings = self.encode_input(df.iloc[start:stop])
                x = torch.cat(num + bin + embeddings, dim=1)
                if layer <= 0:
                    layers = len(self.encoder) + layer
                    x = self.model.encode(x, layers=layers)
                else:
                    x = self.model.encode(x)
                    x = self.model.decode(x, layers=layer)
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

        self.model.eval()
        if self.optim is None:
            self._build_model(df)
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

    def decode_outputs_to_df(self, num, bin, cat):
        """
        Converts the model outputs of the numerical, binary, and categorical features
        back into a pandas dataframe.
        """
        row_count = len(num)
        index = range(row_count)

        num_cols = [x for x in self.numeric_fts.keys()]
        num_df = pd.DataFrame(data=num.cpu().numpy(), index=index)
        num_df.columns = num_cols
        for ft in num_df.columns:
            feature = self.numeric_fts[ft]
            col = num_df[ft]
            trans_col = feature['scaler'].inverse_transform(col.values)
            result = pd.Series(index=index, data=trans_col)
            num_df[ft] = result

        bin_cols = [x for x in self.binary_fts.keys()]
        bin_df = pd.DataFrame(data=bin.cpu().numpy(), index=index)
        bin_df.columns = bin_cols
        bin_df = bin_df.apply(lambda x: round(x)).astype(bool)
        for ft in bin_df.columns:
            feature = self.binary_fts[ft]
            map = {False: feature['cats'][0], True: feature['cats'][1]}
            bin_df[ft] = bin_df[ft].apply(lambda x: map[x])

        cat_df = pd.DataFrame(index=index)
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

        return output_df

    def df_predict(self, df):
        """
        Runs end-to-end model.
        Interprets output and creates a dataframe.
        Outputs dataframe with same shape as input
            containing model predictions.
        """
        self.model.eval()
        data = self.prepare_df(df)
        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            num, bin, cat = self.model(x)
            output_df = self.decode_outputs_to_df(num=num, bin=bin, cat=cat, df=df)

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
        self.model.eval()

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

                num, bin, cat = self.model(input_slice)
                mse_loss_slice: torch.Tensor = self.mse(num, num_target)
                bce_loss_slice: torch.Tensor = self.bce(bin, bin_target)
                # each entry in `cce_loss_slice_of_each_feat` is the cce loss of a feature, ordered by the feature list self.categorical_fts
                cce_loss_slice_of_each_feat = []

                for i, ft in enumerate(self.categorical_fts):
                    loss = self.cce(cat[i], codes[i])
                    # Convert to 2 dimensions
                    cce_loss_slice_of_each_feat.append(loss.data.reshape(-1, 1))
                # merge the tensors into one (n_records * n_features) tensor
                cce_loss_slice = torch.cat(cce_loss_slice_of_each_feat, dim=1)

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
        self.model.eval()

        data = self.prepare_df(df)

        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            num, bin, cat = self.model(x)
            output_df = self.decode_outputs_to_df(num=num, bin=bin, cat=cat)

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
