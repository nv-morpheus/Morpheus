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
import logging
import typing

import dill
import numpy as np
import pandas as pd
import srf
import torch
from dfencoder import AutoEncoder
from srf.core import operators as ops

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.input.cloud_trail_source_stage import CloudTrailSourceStage

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
        self._epochs = epochs
        self._save_model = save_model

        self._model: AutoEncoder = None

    @property
    def model(self):
        return self._model

    def train(self, df: pd.DataFrame) -> AutoEncoder:

        # Determine how much history to save
        if (self._history is not None):
            to_drop = max(len(df) + len(self._history) - self._max_history, 0)

            history = self._history.iloc[to_drop:, :]

            combined_df = pd.concat([history, df])
        else:
            combined_df = df

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
            scaler='gauss_rank',  # feature scaling method
            min_cats=1,  # cut off for minority categories
            progress_bar=False,
        )

        logger.debug("Training AE model for user: '%s'...", self._user_id)
        model.fit(combined_df[combined_df.columns.intersection(self._feature_columns)], epochs=self._epochs)
        logger.debug("Training AE model for user: '%s'... Complete.", self._user_id)

        if (self._save_model):
            self._model = model

        # Save the history for next time
        self._history = combined_df.iloc[max(0, len(combined_df) - self._max_history):, :]

        return model


class TrainAEStage(MultiMessageStage):
    """
    Autoencoder usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    pretrained_filename : str, default = None
        Load a pre-trained model from a file.
    train_data_glob : str, default = None
        Input glob pattern to match files to read.
    train_epochs : int, default = 25
        Passed in as the `epoch` parameter to `AutoEncoder.fit` causes data to be trained in `train_epochs` batches.
    train_max_history : int, default = 1000
        Truncate training data to at most `train_max_history` rows.
    seed : int, default = None
        When not None, ensure random number generators are seeded with `seed` to control reproducibility of user model
        training.
    sort_glob : bool, default = False
        If true the list of files matching `input_glob` will be processed in sorted order.
    """

    def __init__(self,
                 c: Config,
                 pretrained_filename: str = None,
                 train_data_glob: str = None,
                 train_epochs: int = 25,
                 train_max_history: int = 1000,
                 seed: int = None,
                 sort_glob: bool = False):
        super().__init__(c)

        self._config = c
        self._feature_columns = c.ae.feature_columns
        self._batch_size = c.pipeline_batch_size
        self._pretrained_filename = pretrained_filename
        self._train_data_glob: str = train_data_glob
        self._train_epochs = train_epochs
        self._train_max_history = train_max_history
        self._seed = seed
        self._sort_glob = sort_glob

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

    def _get_pretrained_model(self, x: UserMessageMeta):
        return self._pretrained_model

    def _get_per_user_model(self, x: UserMessageMeta):

        if (x.user_id not in self._user_models):
            raise RuntimeError("User ID ({}) was not found in the training dataset and cannot be processed.".format(
                x.user_id))

        return self._user_models[x.user_id].model

    def _train_model(self, x: UserMessageMeta) -> typing.List[MultiAEMessage]:

        if (x.user_id not in self._user_models):
            self._user_models[x.user_id] = _UserModelManager(self._config,
                                                             x.user_id,
                                                             False,
                                                             self._train_epochs,
                                                             self._train_max_history,
                                                             self._seed)

        model = self._user_models[x.user_id].train(x.df)

        return model

    def _build_single(self, seg: srf.Builder, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        get_model_fn = None

        # If a pretrained model was specified, load that now
        if (self._pretrained_filename is not None):
            if (self._train_data_glob is not None):
                logger.warning("Both 'pretrained_filename' and 'train_data_glob' were specified. "
                               "The 'train_data_glob' will be ignored")

            with open(self._pretrained_filename, 'rb') as in_strm:
                self._pretrained_model = dill.load(in_strm)

            get_model_fn = self._get_pretrained_model

        elif (self._train_data_glob is not None):
            file_list = glob.glob(self._train_data_glob)
            if self._sort_glob:
                file_list = sorted(file_list)

            user_to_df = CloudTrailSourceStage.files_to_dfs_per_user(file_list,
                                                                     FileTypes.Auto,
                                                                     self._config.ae.userid_column_name,
                                                                     self._feature_columns,
                                                                     self._config.ae.userid_filter)

            for user_id, df in user_to_df.items():
                self._user_models[user_id] = _UserModelManager(self._config,
                                                               user_id,
                                                               True,
                                                               self._train_epochs,
                                                               self._train_max_history,
                                                               self._seed)

                self._user_models[user_id].train(df)

            get_model_fn = self._get_per_user_model

        else:
            get_model_fn = self._train_model

        def node_fn(input: srf.Observable, output: srf.Subscriber):

            def on_next(x: UserMessageMeta):

                model = get_model_fn(x)

                full_message = MultiAEMessage(meta=x, mess_offset=0, mess_count=x.count, model=model)

                to_send = []

                # Now split into batches
                for i in range(0, full_message.mess_count, self._batch_size):

                    to_send.append(full_message.get_slice(i, min(i + self._batch_size, full_message.mess_count)))

                return to_send

            input.pipe(ops.map(on_next), ops.flatten()).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node

        return stream, MultiAEMessage
