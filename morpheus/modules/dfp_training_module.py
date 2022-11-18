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

import logging
import typing

import srf
from dfencoder import AutoEncoder
from srf.core import operators as ops

from morpheus.modules.module import Module
from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from examples.digital_fingerprinting.production.morpheus.dfp.messages.multi_dfp_message import MultiDFPMessage

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTrainingModule(Module):

    def __init__(self,
                 config: typing.Dict = {},
                 version: typing.List = [22, 11, 0],
                 module_id: str = "DFPTraining",
                 module_name: str = "DFPTrainingModule",
                 module_namespace: str = "DFP",
                 model_kwargs: dict = None):

        super().__init__(config, version, module_id, module_name, module_namespace)

        self._model_kwargs = {
            "encoder_layers": [512, 500],  # layers of the encoding part
            "decoder_layers": [512],  # layers of the decoding part
            "activation": 'relu',  # activation function
            "swap_p": 0.2,  # noise parameter
            "lr": 0.001,  # learning rate
            "lr_decay": .99,  # learning decay
            "batch_size": 512,
            "verbose": False,
            "optimizer": 'sgd',  # SGD optimizer is selected(Stochastic gradient descent)
            "scaler": 'standard',  # feature scaling method
            "min_cats": 1,  # cut off for minority categories
            "progress_bar": False,
            "device": "cuda"
        }

        # Update the defaults
        self._model_kwargs.update(model_kwargs if model_kwargs is not None else {})

    def on_data(self, message):
        if (message is None or message.mess_count == 0):
            return None

        user_id = message.user_id

        model = AutoEncoder(**self._model_kwargs)

        final_df = message.get_meta_dataframe()

        # Only train on the feature columns
        final_df = final_df[final_df.columns.intersection(self._config.ae.feature_columns)]

        logger.debug("Training AE model for user: '%s'...", user_id)
        model.fit(final_df, epochs=30)
        logger.debug("Training AE model for user: '%s'... Complete.", user_id)

        output_message = MultiAEMessage(message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=model)

        return output_message
