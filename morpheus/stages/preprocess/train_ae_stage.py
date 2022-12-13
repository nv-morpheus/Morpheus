# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import glob
import importlib
import logging
import pathlib
import typing

import dill
import mrc
import numpy as np
import pandas as pd
import torch
from dfencoder import AutoEncoder
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class _UserModelManager(object):

    def __init__(self,
                 c: Config,
                 user_id: str,
                 save_model: bool,
                 epochs: int,
                 max_history: int,
                 seed: int = None) -> None:
        super().__init__()

        self._user_id = user_id
        self._history: pd.DataFrame = None
        self._max_history: int = max_history
        self._seed: int = seed
        self._feature_columns = c.ae.feature_columns
        self._feature_scaler = c.ae.feature_scaler
        self._epochs = epochs
        self._save_model = save_model

        self._model: AutoEncoder = None
        self._train_scores_mean = None
        self._train_scores_std = None

    @property
    def model(self):
        return self._model

    @property
    def train_scores_mean(self):
        return self._train_scores_mean

    @property
    def train_scores_std(self):
        return self._train_scores_std

    def train(self, df: pd.DataFrame) -> AutoEncoder:

        # Determine how much history to save
        if (self._history is not None):
            to_drop = max(len(df) + len(self._history) - self._max_history, 0)

            history = self._history.iloc[to_drop:, :]

            train_df = pd.concat([history, df])
        else:
            train_df = df

        # If the seed is set, enforce that here
        if (self._seed is not None):
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed(self._seed)
            np.random.seed(self._seed)
            torch.backends.cudnn.deterministic = True

        model = AutoEncoder(
            encoder_layers=[512, 500],  # layers of the encoding part
            decoder_layers=[512],  # layers of the decoding part
            activation='relu',  # activation function
            swap_p=0.2,  # noise parameter
            lr=0.01,  # learning rate
            lr_decay=.99,  # learning decay
            batch_size=512,
            # logger='ipynb',
            verbose=False,
            optimizer='sgd',  # SGD optimizer is selected(Stochastic gradient descent)
            scaler=self._feature_scaler,  # feature scaling method
            min_cats=1,  # cut off for minority categories
            progress_bar=False)

        logger.debug("Training AE model for user: '%s'...", self._user_id)
        model.fit(train_df, epochs=self._epochs)
        train_loss_scores = model.get_anomaly_score(train_df)
        scores_mean = train_loss_scores.mean()
        scores_std = train_loss_scores.std()

        logger.debug("Training AE model for user: '%s'... Complete.", self._user_id)

        if (self._save_model):
            self._model = model
            self._train_scores_mean = scores_mean
            self._train_scores_std = scores_std

        # Save the history for next time
        self._history = train_df.iloc[max(0, len(train_df) - self._max_history):, :]

        return model, scores_mean, scores_std


@register_stage("train-ae", modes=[PipelineModes.AE])
class TrainAEStage(MultiMessageStage):
    """
    Train an Autoencoder model on incoming data.

    This stage is used to train an Autoencoder model on incoming data a supply that model to downstream stages. The
    Autoencoder workflows use this stage as a pre-processing step to build the model for inference.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    pretrained_filename : pathlib.Path, default = None
        Loads a single pre-trained model for all users.
    train_data_glob : str, default = None
        On startup, all files matching this glob pattern will be loaded and used to train a model for each unique user
        ID.
    source_stage_class : str, default = None
        If train_data_glob provided, use source stage to batch training data per user.
    train_epochs : int, default = 25, min = 1
        The number of epochs to train user models for. Passed in as the `epoch` parameter to `AutoEncoder.fit` causes
        data to be trained in `train_epochs` batches.
    train_min_history : int, default = 300
        Minimum number of rows to train user model.
    train_max_history : int, default = 1000, min = 1
        Maximum amount of rows that will be retained in history. As new data arrives, models will be retrained with a
        maximum number of rows specified by this value.
    seed : int, default = None
        Seed to use when training. When not None, ensure random number generators are seeded with `seed` to control
        reproducibility of user model training.
    sort_glob : bool, default = False, is_flag = True
        If true the list of files matching `input_glob` will be processed in sorted order.
    models_output_filename : pathlib.Path, default = None, writable = True
        The location to write trained models to.
    """

    def __init__(self,
                 c: Config,
                 pretrained_filename: pathlib.Path = None,
                 train_data_glob: str = None,
                 source_stage_class: str = None,
                 train_epochs: int = 25,
                 train_min_history: int = 300,
                 train_max_history: int = 1000,
                 seed: int = None,
                 sort_glob: bool = False,
                 models_output_filename: pathlib.Path = None):
        super().__init__(c)

        self._config = c
        self._feature_columns = c.ae.feature_columns
        self._use_generic_model = c.ae.use_generic_model
        self._batch_size = c.pipeline_batch_size
        self._pretrained_filename = pretrained_filename
        self._train_data_glob: str = train_data_glob
        self._train_epochs = train_epochs
        self._train_min_history = train_min_history
        self._train_max_history = train_max_history
        self._seed = seed
        self._sort_glob = sort_glob
        self._models_output_filename = models_output_filename

        self._source_stage_class = source_stage_class
        if self._source_stage_class is not None:
            source_stage_module, source_stage_classname = self._source_stage_class.rsplit('.', 1)
            # load the source stage module, will raise ImportError if module cannot be loaded
            source_stage_module = importlib.import_module(source_stage_module)
            # get the source stage class, will raise AttributeError if class cannot be found
            self._source_stage_class = getattr(source_stage_module, source_stage_classname)

        # Single model for the entire pipeline
        self._pretrained_model: AutoEncoder = None

        # Per user model data
        self._user_models: typing.Dict[str, _UserModelManager] = {}

    @property
    def name(self) -> str:
        return "train-ae"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (UserMessageMeta, )

    def supports_cpp_node(self):
        return False

    def _get_per_user_model(self, x: UserMessageMeta):

        model = None
        train_scores_mean = None
        train_scores_std = None
        user_model = None

        if x.user_id in self._user_models:
            user_model = self._user_models[x.user_id]
        elif self._use_generic_model and "generic" in self._user_models.keys():
            user_model = self._user_models["generic"]

        if (user_model is not None):
            model = user_model.model
            train_scores_mean = user_model.train_scores_mean
            train_scores_std = user_model.train_scores_std

        return model, train_scores_mean, train_scores_std

    def _train_model(self, x: UserMessageMeta) -> typing.List[MultiAEMessage]:

        if (x.user_id not in self._user_models):
            self._user_models[x.user_id] = _UserModelManager(self._config,
                                                             x.user_id,
                                                             False,
                                                             self._train_epochs,
                                                             self._train_max_history,
                                                             self._seed)

        return self._user_models[x.user_id].train(x.df)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        get_model_fn = None

        # If a pretrained model was specified, load that now
        if (self._pretrained_filename is not None):
            if (self._train_data_glob is not None):
                logger.warning("Both 'pretrained_filename' and 'train_data_glob' were specified. "
                               "The 'train_data_glob' will be ignored")

            with open(self._pretrained_filename, 'rb') as in_strm:
                # self._pretrained_model = dill.load(in_strm)
                self._user_models = dill.load(in_strm)

            # get_model_fn = self._get_pretrained_model
            get_model_fn = self._get_per_user_model

        elif (self._train_data_glob is not None):
            if (self._source_stage_class is None):
                raise RuntimeError("source_stage_class must be provided with train_data_glob")
            file_list = glob.glob(self._train_data_glob)
            if self._sort_glob:
                file_list = sorted(file_list)

            user_to_df = self._source_stage_class.files_to_dfs_per_user(file_list,
                                                                        self._config.ae.userid_column_name,
                                                                        self._feature_columns,
                                                                        self._config.ae.userid_filter)

            if self._use_generic_model:
                self._user_models["generic"] = _UserModelManager(self._config,
                                                                 "generic",
                                                                 True,
                                                                 self._train_epochs,
                                                                 self._train_max_history,
                                                                 self._seed)

                all_users_df = pd.concat(user_to_df.values(), ignore_index=True)
                all_users_df = self._source_stage_class.derive_features(all_users_df, self._feature_columns)
                all_users_df = all_users_df.fillna("nan")
                self._user_models["generic"].train(all_users_df)

            for user_id, df in user_to_df.items():
                if len(df.index) >= self._train_min_history:
                    self._user_models[user_id] = _UserModelManager(self._config,
                                                                   user_id,
                                                                   True,
                                                                   self._train_epochs,
                                                                   self._train_max_history,
                                                                   self._seed)

                    # Derive features here
                    # print(df)
                    df = self._source_stage_class.derive_features(df, self._feature_columns)
                    df = df.fillna("nan")
                    self._user_models[user_id].train(df)

            # Save trained user models
            if self._models_output_filename is not None:
                with open(self._models_output_filename, 'wb') as out_strm:
                    dill.dump(self._user_models, out_strm)

            get_model_fn = self._get_per_user_model

        else:
            get_model_fn = self._train_model

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            def on_next(x: UserMessageMeta):

                model, scores_mean, scores_std = get_model_fn(x)

                full_message = MultiAEMessage(meta=x,
                                              mess_offset=0,
                                              mess_count=x.count,
                                              model=model,
                                              train_scores_mean=scores_mean,
                                              train_scores_std=scores_std)

                to_send = []

                # Now split into batches
                for i in range(0, full_message.mess_count, self._batch_size):

                    to_send.append(full_message.get_slice(i, min(i + self._batch_size, full_message.mess_count)))

                return to_send

            obs.pipe(ops.map(on_next), ops.flatten()).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(stream, node)
        stream = node

        return stream, MultiAEMessage
