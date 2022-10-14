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

import logging
import typing

import numpy as np
import pandas as pd
import srf

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.config import PipelineModes
from morpheus.messages import MessageMeta

logger = logging.getLogger(__name__)


@register_stage("dfp-viz-postproc", modes=[PipelineModes.AE])
class DFPVizPostprocStage(SinglePortStage):

    def __init__(self, c: Config, period: str ="D"):
        super().__init__(c)

        self._user_column_name = c.ae.userid_column_name
        self._timestamp_column = c.ae.timestamp_column_name
        self._feature_columns = c.ae.feature_columns
        self._period = period

    @property
    def name(self) -> str:
        return "dfp-viz-postproc"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[`morpheus.pipeline.messages.MultiAEMessage`, ]
            Accepted input types.

        """
        return (MultiAEMessage, )

    def supports_cpp_node(self):
        return False

    def _postprocess(self, x: MultiAEMessage):

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min())

        viz_pdf = pd.DataFrame()
        viz_pdf["user"] = x.get_meta(self._user_column_name)
        viz_pdf["time"] = x.get_meta(self._timestamp_column)

        log_mean = x.get_meta("mean_abs_z").apply(np.log1p)
        viz_pdf["anomalyScore"] = normalize(log_mean)
        viz_pdf["anomalyScore_mean"] = viz_pdf["anomalyScore"].mean()

        for f in self._feature_columns:
            log_f_loss = x.get_meta(f + "_z_loss").apply(np.log1p)
            viz_pdf[f + "_score"] =  normalize(log_f_loss)
            viz_pdf[f + "_score_mean"] = viz_pdf[f + "_score"].mean()

        return MessageMeta(df=viz_pdf)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        stream = builder.make_node(self.unique_name, self._postprocess)

        builder.make_edge(input_stream[0], stream)

        return stream, MessageMeta
