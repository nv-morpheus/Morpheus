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
import time
import typing
from datetime import datetime

import numpy as np
import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import DFPMessageMeta

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPPostprocessingStage(SinglePortStage):

    def __init__(self, c: Config, z_score_threshold=2.0):
        super().__init__(c)

        self._z_score_threshold = z_score_threshold

    @property
    def name(self) -> str:
        return "dfp-postproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiAEMessage, )

    def _extract_events(self, message: MultiAEMessage):

        z_scores = message.get_meta("mean_abs_z")

        above_threshold_df = message.get_meta()[z_scores > self._z_score_threshold]

        if (not above_threshold_df.empty):
            above_threshold_df['event_time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            above_threshold_df = above_threshold_df.replace(np.nan, 'NaN', regex=True)

            return above_threshold_df

        return None

    def on_data(self, message: MultiAEMessage):
        if (not message):
            return None

        start_time = time.time()

        extracted_events = self._extract_events(message)

        duration = (time.time() - start_time) * 1000.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Completed postprocessing for user %s in %s ms. Event count: %s. Start: %s, End: %s",
                         message.meta.user_id,
                         duration,
                         0 if extracted_events is None else len(extracted_events),
                         message.get_meta(self._config.ae.timestamp_column_name).min(),
                         message.get_meta(self._config.ae.timestamp_column_name).max())

        if (extracted_events is None):
            return None

        return DFPMessageMeta(extracted_events, user_id=message.meta.user_id)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, DFPMessageMeta
