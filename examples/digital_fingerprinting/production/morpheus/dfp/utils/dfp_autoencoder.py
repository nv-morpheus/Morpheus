# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict

import dfencoder
import numpy as np
import pandas as pd
import torch


class FixedStandardScalar(dfencoder.StandardScaler):

    def fit(self, x):
        super().fit(x)

        # Having a std == 0 (when all values are the same), breaks training. Just use 1.0 in this case
        if (self.std == 0.0):
            self.std = 1.0


class DFPAutoEncoder(dfencoder.AutoEncoder):

    def __init__(self,
                 encoder_layers=None,
                 decoder_layers=None,
                 encoder_dropout=None,
                 decoder_dropout=None,
                 encoder_activations=None,
                 decoder_activations=None,
                 activation='relu',
                 min_cats=10,
                 swap_p=0.15,
                 lr=0.01,
                 batch_size=256,
                 eval_batch_size=1024,
                 optimizer='adam',
                 amsgrad=False,
                 momentum=0,
                 betas=...,
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
                 *args,
                 **kwargs):
        super().__init__(encoder_layers,
                         decoder_layers,
                         encoder_dropout,
                         decoder_dropout,
                         encoder_activations,
                         decoder_activations,
                         activation,
                         min_cats,
                         swap_p,
                         lr,
                         batch_size,
                         eval_batch_size,
                         optimizer,
                         amsgrad,
                         momentum,
                         betas,
                         dampening,
                         weight_decay,
                         lr_decay,
                         nesterov,
                         verbose,
                         device,
                         logger,
                         logdir,
                         project_embeddings,
                         run,
                         progress_bar,
                         n_megabatches,
                         scaler,
                         *args,
                         **kwargs)

        self.cyclical_fts = OrderedDict()
        self.feature_loss_stats = dict()
        self.val_loss_mean = None
        self.val_loss_std = None

    def get_scaler(self, name):
        scaler_result = super().get_scaler(name)

        # Use the fixed scalar instead of the standard
        if (scaler_result == dfencoder.StandardScaler):
            return FixedStandardScalar

        return scaler_result

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
        objects = list(df.select_dtypes(include=["object", "string"]).columns)
        for ft in objects:
            feature = {}
            vl = df[ft].value_counts()
            cats = list(vl[vl >= self.min_cats].index)
            feature['cats'] = cats
            self.categorical_fts[ft] = feature

    def _create_stat_dict(self, a):
        scaler = dfencoder.StandardScaler()
        scaler.fit(a)
        mean = scaler.mean
        std = scaler.std
        return {'scaler': scaler, 'mean': mean, 'std': std}

    def fit(self, df, epochs=1, val=None):
        """Does training."""
        pdf = df.copy()
        # if val is None:
        #     pdf_val = None
        # else:
        #     pdf_val = val.copy()

        if self.optim is None:
            self.build_model(df)
        if self.n_megabatches == 1:
            df = self.prepare_df(df)

        if val is not None:
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
        for i in range(epochs):
            self.train()
            if self.verbose:
                print(f'training epoch {i + 1}...')
            df = df.sample(frac=1.0)
            df = dfencoder.EncoderDataFrame(df)
            if self.n_megabatches > 1:
                self.train_megabatch_epoch(n_updates, df)
            else:
                input_df = df.swap(likelihood=self.swap_p)
                self.train_epoch(n_updates, input_df, df)

            if self.lr_decay is not None:
                self.lr_decay.step()

            if val is not None:
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

        # Getting training loss statistics
        # mse_loss, bce_loss, cce_loss, _ = self.get_anomaly_score(pdf) if pdf_val is None else self.get_anomaly_score(pd.concat([pdf, pdf_val]))
        mse_loss, bce_loss, cce_loss, _ = self.get_anomaly_score(pdf)
        for i, ft in enumerate(self.numeric_fts):
            i_loss = mse_loss[:, i].cpu().numpy()
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)
        for i, ft in enumerate(self.binary_fts):
            i_loss = bce_loss[:, i].cpu().numpy()
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)
        for i, ft in enumerate(self.categorical_fts):
            i_loss = cce_loss[i].cpu().numpy()
            self.feature_loss_stats[ft] = self._create_stat_dict(i_loss)

    def get_anomaly_score(self, df):
        """
        Returns a per-row loss of the input dataframe.
        Does not corrupt inputs.
        """
        self.eval()
        data = self.prepare_df(df)
        input = self.build_input_tensor(data)

        num_target, bin_target, codes = self.compute_targets(data)

        with torch.no_grad():
            num, bin, cat = self.forward(input)

        mse_loss = self.mse(num, num_target)
        net_loss = [mse_loss.data]
        bce_loss = self.bce(bin, bin_target)
        net_loss += [bce_loss.data]
        cce_loss = []
        for i, ft in enumerate(self.categorical_fts):
            loss = self.cce(cat[i], codes[i])
            cce_loss.append(loss)
            net_loss += [loss.data.reshape(-1, 1)]

        net_loss = torch.cat(net_loss, dim=1).mean(dim=1)
        return mse_loss, bce_loss, cce_loss, net_loss.cpu().numpy()

    def get_scaled_anomaly_scores(self, df):
        self.eval()
        data = self.prepare_df(df)
        input = self.build_input_tensor(data)

        num_target, bin_target, codes = self.compute_targets(data)
        with torch.no_grad():
            num, bin, cat = self.forward(input)

        mse_loss = self.mse(num, num_target)
        mse_scaled = torch.zeros(mse_loss.shape)
        for i, ft in enumerate(self.numeric_fts):
            mse_scaled[:, i] = torch.tensor(self.feature_loss_stats[ft]['scaler'].transform(mse_loss[:,
                                                                                                     i].cpu().numpy()))
        bce_loss = self.bce(bin, bin_target)
        bce_scaled = torch.zeros(bce_loss.shape)
        for i, ft in enumerate(self.binary_fts):
            bce_scaled[:, i] = torch.tensor(self.feature_loss_stats[ft]['scaler'].transform(mse_loss[:,
                                                                                                     i].cpu().numpy()))
        cce_scaled = []
        for i, ft in enumerate(self.categorical_fts):
            loss = torch.tensor(self.feature_loss_stats[ft]['scaler'].transform(
                self.cce(cat[i], codes[i]).cpu().numpy()))
            cce_scaled.append(loss)

        return mse_scaled, bce_scaled, cce_scaled

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
            # get argmax excluding NaN column (impute with next-best guess)
            codes = torch.argmax(cat[i], dim=1).cpu().numpy()
            cat_df[ft] = pd.Categorical.from_codes(codes, categories=feature['cats'] + ["_other"])
            cat_df[ft] = cat_df[ft].astype("string")

        # concat
        output_df = pd.concat([num_df, bin_df, cat_df], axis=1)

        return output_df

    def get_results(self, df, return_abs=False):
        pdf = df.copy()
        self.eval()
        data = self.prepare_df(df)
        orig_cols = data.columns
        with torch.no_grad():
            num, bin, embeddings = self.encode_input(data)
            x = torch.cat(num + bin + embeddings, dim=1)
            x = self.encode(x)
            output_df = self.decode_to_df(x, df=df)
        mse, bce, cce, _ = self.get_anomaly_score(df)
        mse_scaled, bce_scaled, cce_scaled = self.get_scaled_anomaly_scores(df)
        for i, ft in enumerate(self.numeric_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = mse[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = mse_scaled[:, i].cpu().numpy() if not return_abs else abs(mse_scaled[:,
                                                                                                       i].cpu().numpy())
        for i, ft in enumerate(self.binary_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = bce[:, i].cpu().numpy()
            pdf[ft + '_z_loss'] = bce_scaled[:, i].cpu().numpy() if not return_abs else abs(bce_scaled[:,
                                                                                                       i].cpu().numpy())
        for i, ft in enumerate(self.categorical_fts):
            pdf[ft + '_pred'] = output_df[ft]
            pdf[ft + '_loss'] = cce[i].cpu().numpy()
            pdf[ft + '_z_loss'] = cce_scaled[i].cpu().numpy() if not return_abs else abs(cce_scaled[i].cpu().numpy())
        all_cols = [[c, c + '_pred', c + '_loss', c + '_z_loss'] for c in orig_cols]
        result_cols = [col for col_collection in all_cols for col in col_collection]
        z_losses = [c + '_z_loss' for c in orig_cols]
        pdf['max_abs_z'] = pdf[z_losses].max(axis=1)
        pdf['mean_abs_z'] = pdf[z_losses].mean(axis=1)
        result_cols.append('max_abs_z')
        result_cols.append('mean_abs_z')
        return pdf[result_cols]
