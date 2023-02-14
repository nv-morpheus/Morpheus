# Copyright (c) 2023, NVIDIA CORPORATION.
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
import time
from datetime import datetime

import mrc
import numpy as np
from mrc.core import operators as ops

from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.utils.module_ids import MODULE_NAMESPACE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

from ..utils.module_ids import DFP_POST_PROCESSING

logger = logging.getLogger(__name__)


@register_module(DFP_POST_PROCESSING, MODULE_NAMESPACE)
def dfp_postprocessing(builder: mrc.Builder):
    """
    Postprocessing module function.

    Parameters
    ----------
    builder : mrc.Builder
        Pipeline budler instance.
    """

    config = get_module_config(DFP_POST_PROCESSING, builder)

    timestamp_column_name = config.get("timestamp_column_name", None)

    def process_events(message: MultiAEMessage):
        # Assume that a filter stage preceedes this stage
        df = message.get_meta()
        df['event_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        df.replace(np.nan, 'NaN', regex=True, inplace=True)
        message.set_meta(None, df)

    def on_data(message: MultiAEMessage):
        if (not message or message.mess_count == 0):
            return None

        start_time = time.time()

        process_events(message)

        duration = (time.time() - start_time) * 1000.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Completed postprocessing for user %s in %s ms. Event count: %s. Start: %s, End: %s",
                         message.meta.user_id,
                         duration,
                         message.mess_count,
                         message.get_meta(timestamp_column_name).min(),
                         message.get_meta(timestamp_column_name).max())

        return message

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

    node = builder.make_node_full(DFP_POST_PROCESSING, node_fn)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
