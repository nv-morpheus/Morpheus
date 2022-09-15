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

import pandas as pd
import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from ..messages.multi_dfp_message import DFPMessageMeta

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPS3BatcherStage(SinglePortStage):

    def __init__(self, c: Config, date_conversion_func, period="D", sampling_rate_s=0):
        super().__init__(c)

        self._date_conversion_func = date_conversion_func
        self._sampling_rate_s = sampling_rate_s
        self._period = period

    @property
    def name(self) -> str:
        return "dfp-s3-batcher"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (typing.Any, )

    def on_data(self, s3_objects: typing.Union[typing.Any, typing.List[typing.Any]]):
        s3_object_list = s3_objects
        if (not isinstance(s3_object_list, list)):
            # Convert to a list
            s3_object_list = [s3_object_list]

        # Create a dataframe with the incoming metadata
        if ((len(s3_object_list) > 1) and (self._sampling_rate_s > 0)):
            s3_sampled_list = []
            s3_object_list.sort(key=lambda s3_object: self._date_conversion_func(s3_object))
            ts_last = self._date_conversion_func(s3_object_list[0])
            s3_sampled_list.append(s3_object_list[0])
            for idx in range(1, len(s3_object_list)):
                ts = self._date_conversion_func(s3_object_list[idx])
                if ((ts - ts_last).seconds >= self._sampling_rate_s):
                    s3_sampled_list.append(s3_object_list[idx])
                    ts_last = ts
            else:
                s3_object_list = s3_sampled_list

        df = pd.DataFrame()

        df["dfp_timestamp"] = [self._date_conversion_func(s3_object) for s3_object in s3_object_list]
        df["key"] = [s3_object.key for s3_object in s3_object_list]
        df["objects"] = s3_object_list

        # Now split by the batching settings
        df_period = df["dfp_timestamp"].dt.to_period(self._period)

        period_gb = df.groupby(df_period)

        output_batches = []

        n_groups = len(period_gb)
        for group in period_gb.groups:
            period_df = period_gb.get_group(group)
            obj_list = period_df["objects"].to_list()
            obj_list = [(s3_object, n_groups) for s3_object in obj_list]

            output_batches.append(obj_list)

        return output_batches

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.flatten()).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, DFPMessageMeta
