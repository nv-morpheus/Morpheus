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

import fsspec
import pandas as pd
import srf
from srf.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPFileBatcherStage(SinglePortStage):

    def __init__(self, c: Config, date_conversion_func, period="D", sampling_rate_s=0, start_time=None, end_time=None):
        super().__init__(c)

        self._date_conversion_func = date_conversion_func
        self._sampling_rate_s = sampling_rate_s
        self._period = period

    @property
    def name(self) -> str:
        return "dfp-file-batcher"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (fsspec.core.OpenFiles, )

    def on_data(self, file_objects: fsspec.core.OpenFiles):

        file_object_list = file_objects

        # Create a dataframe with the incoming metadata
        if ((len(file_object_list) > 1) and (self._sampling_rate_s > 0)):
            file_sampled_list = []

            file_object_list.sort(key=lambda file_object: self._date_conversion_func(file_object))

            ts_last = self._date_conversion_func(file_object_list[0])

            file_sampled_list.append(file_object_list[0])

            for idx in range(1, len(file_object_list)):
                ts = self._date_conversion_func(file_object_list[idx])

                if ((ts - ts_last).seconds >= self._sampling_rate_s):

                    file_sampled_list.append(file_object_list[idx])
                    ts_last = ts
            else:
                file_object_list = file_sampled_list

        df = pd.DataFrame()

        df["dfp_timestamp"] = [self._date_conversion_func(file_object) for file_object in file_object_list]
        df["key"] = [file_object.full_name for file_object in file_object_list]
        df["objects"] = file_object_list

        # Now split by the batching settings
        df_period = df["dfp_timestamp"].dt.to_period(self._period)

        period_gb = df.groupby(df_period)

        output_batches = []

        n_groups = len(period_gb)
        for group in period_gb.groups:
            period_df = period_gb.get_group(group)

            obj_list = fsspec.core.OpenFiles(period_df["objects"].to_list(), mode=file_objects.mode, fs=file_objects.fs)

            output_batches.append((obj_list, n_groups))

        return output_batches

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.flatten()).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, typing.List[fsspec.core.OpenFiles]
