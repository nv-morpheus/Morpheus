# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
"""Training stage for the DFP pipeline."""

import logging
import typing

import mrc
from mrc.core import operators as ops
from sklearn.model_selection import train_test_split

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.models.dfencoder import AutoEncoder
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import MultiDFPMessage

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPTraining(SinglePortStage):
    """
    Performs training of the DFP model using `AutoEncoder`. The trained model is then attached to output messages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    model_kwargs : dict
        Keyword arguments to pass to the `AutoEncoder` constructor.
    epochs : int
        Number of epochs to train the model for.
    validation_size : float
        Fraction of the training data to use for validation. Must be in the (0, 1) range.
    """

    def __init__(self, c: Config, model_kwargs: dict = None, epochs=30, validation_size=0.0):
        super().__init__(c)

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

        self._epochs = epochs

        if (validation_size >= 0.0 and validation_size < 1.0):
            self._validation_size = validation_size
        else:
            raise ValueError(f"validation_size={validation_size} should be a positive float in the (0, 1) range")

    @property
    def name(self) -> str:
        """Stage name."""
        return "dfp-training"

    def supports_cpp_node(self):
        """Whether this stage supports a C++ node."""
        return False

    def accepted_types(self) -> typing.Tuple:
        """Indicate which input message types this stage accepts."""
        return (MultiDFPMessage, )

    def on_data(self, message: MultiDFPMessage) -> MultiAEMessage:
        """Train the model and attach it to the output message."""
        if (message is None or message.mess_count == 0):
            return None

        user_id = message.user_id

        model = AutoEncoder(**self._model_kwargs)

        train_df = message.get_meta_dataframe()

        # Only train on the feature columns
        train_df = train_df[train_df.columns.intersection(self._config.ae.feature_columns)]
        validation_df = None
        run_validation = False

        # Split into training and validation sets
        if self._validation_size > 0.0:
            train_df, validation_df = train_test_split(train_df, test_size=self._validation_size, shuffle=False)
            run_validation = True

        logger.debug("Training AE model for user: '%s'...", user_id)
        model.fit(train_df, epochs=self._epochs, val_data=validation_df, run_validation=run_validation)
        logger.debug("Training AE model for user: '%s'... Complete.", user_id)

        output_message = MultiAEMessage(meta=message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=model)

        return output_message

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = builder.make_node(self.unique_name, ops.map(self.on_data), ops.filter(lambda x: x is not None))
        builder.make_edge(input_stream[0], stream)

        return stream, MultiAEMessage
