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
import pickle
import time

import mrc
from mrc.core import operators as ops

from morpheus.utils.column_info import process_dataframe
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module
from morpheus.messages.message_control import MessageControl

from ..messages.multi_dfp_message import MultiDFPMessage
from ..utils.module_ids import DFP_DATA_PREP

logger = logging.getLogger(__name__)


@register_module(DFP_DATA_PREP, MODULE_NAMESPACE)
def dfp_data_prep(builder: mrc.Builder) -> MessageControl:
    """
    This module function prepares data for either inference or model training.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_DATA_PREP, builder)

    schema_config = config.get("schema", None)
    schema_str = schema_config.get("schema_str", None)
    encoding = schema_config.get("encoding", None)
    timestamp_column_name = config.get("timestamp_column_name", None)

    schema = pickle.loads(bytes(schema_str, encoding))

    def process_features(message: MultiDFPMessage):

        if (message is None):
            return None

        start_time = time.time()

        # Process the columns
        df_processed = process_dataframe(message.get_meta_dataframe(), schema)

        message.set_meta_dataframe(list(df_processed.columns), df_processed)

        if logger.isEnabledFor(logging.DEBUG):
            duration = (time.time() - start_time) * 1000.0

            logger.debug("Preprocessed %s data for logs in %s to %s in %s ms",
                         message.mess_count,
                         message.get_meta(timestamp_column_name).min(),
                         message.get_meta(timestamp_column_name).max(),
                         duration)
        message_config = {
            "tasks": [{
                "type": "inference",
                "params": {
                    "user_id": message.user_id,
                    "data": "payload",
                    "mess_offset": message.mess_offset,
                    "mess_count": message.mess_count
                }
            }]
        }

        control_message = MessageControl(message_config)
        control_message.payload(message.meta)

        return control_message

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(process_features)).subscribe(sub)

    node = builder.make_node_full(DFP_DATA_PREP, node_fn)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
