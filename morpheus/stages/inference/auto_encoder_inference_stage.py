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

import typing

import numpy as np
import pandas as pd
import srf

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("inf-pytorch", modes=[PipelineModes.AE])
class AutoEncoderInferenceStage(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)
        self._feature_columns = c.ae.feature_columns

    @property
    def name(self) -> str:
        return "inference-ae"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiAEMessage, )

    def on_data(self, message: MultiAEMessage):
        if (not message or message.mess_count == 0):
            return None

        data = message.get_meta(message.meta.df.columns.intersection(self._feature_columns))
        data = data.fillna("nan")
        autoencoder = message.model

        pred_cols = [x + "_pred" for x in self._feature_columns]
        loss_cols = [x + "_loss" for x in self._feature_columns]
        z_loss_cols = [x + "_z_loss" for x in self._feature_columns]
        abs_z_cols = ["max_abs_z", "mean_abs_z"]
        results_cols = pred_cols + loss_cols + z_loss_cols + abs_z_cols
        results_df = pd.DataFrame(np.empty((len(data), (3 * len(self._feature_columns) + 2)), dtype=object),
                                  columns=results_cols)

        output_message = MultiAEMessage(message.meta,
                                        mess_offset=message.mess_offset,
                                        mess_count=message.mess_count,
                                        model=autoencoder)

        if autoencoder is not None:
            ae_results = autoencoder.get_results(data, return_abs=True)
            for col in ae_results:
                results_df[col] = ae_results[col]

        output_message.set_meta(list(results_df.columns), results_df)

        return output_message

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:
        node = builder.make_node(self.unique_name, self.on_data)
        builder.make_edge(input_stream[0], node)

        return node, MultiAEMessage
