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

import logging

import mrc
from dfencoder import AutoEncoder
from mrc.core import operators as ops

from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.module_ids import DFP_TRAINING

logger = logging.getLogger(__name__)


@register_module(DFP_TRAINING, MODULE_NAMESPACE)
def dfp_training(builder: mrc.Builder):
    """
    Model training is done using this module function.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_TRAINING, builder)

    feature_columns = config.get("feature_columns", None)
    validation_size = config.get("validation_size", 0.0)
    epochs = config.get("epochs", None)
    model_kwargs = config.get("model_kwargs", None)

    if (validation_size > 0.0 and validation_size < 1.0):
        validation_size = validation_size
    else:
        raise ValueError("validation_size={0} should be a positive float in the "
                         "(0, 1) range".format(validation_size))

    def on_data(message: MultiDFPMessage):
        if (message is None or message.mess_count == 0):
            return None

        user_id = message.user_id

        model = AutoEncoder(**model_kwargs)

        final_df = message.get_meta_dataframe()

        # Only train on the feature columns
        final_df = final_df[final_df.columns.intersection(feature_columns)]

        logger.debug("Training AE model for user: '%s'...", user_id)
        model.fit(final_df, epochs=epochs)
        logger.debug("Training AE model for user: '%s'... Complete.", user_id)

        output_message = MultiAEMessage(message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=model)

        return output_message

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node_full(DFP_TRAINING, node_fn)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
