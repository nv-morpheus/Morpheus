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
import typing
from collections import namedtuple
from datetime import datetime

import fsspec
import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.config import Config
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger("morpheus.{}".format(__name__))

TimestampFileObj = namedtuple("TimestampFileObj", ["timestamp", "file_object"])


class DFPFileBatcherStage(SinglePortStage):

    def __init__(self,
                 c: Config,
                 date_conversion_func,
                 period="D",
                 sampling_rate_s=0,
                 start_time: datetime = None,
                 end_time: datetime = None):
        super().__init__(c)

        self._date_conversion_func = date_conversion_func
        self._sampling_rate_s = sampling_rate_s
        self._period = period
        self._start_time = start_time
        self._end_time = end_time

    @property
    def name(self) -> str:
        return "dfp-file-batcher"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (fsspec.core.OpenFiles, )

    def on_data(self, file_objects: fsspec.core.OpenFiles):

        # Determine the date of the file, and apply the window filter if we have one
        ts_and_files = []
        for file_object in file_objects:
            ts = self._date_conversion_func(file_object)

            # Exclude any files outside the time window
            if ((self._start_time is not None and ts < self._start_time)
                    or (self._end_time is not None and ts > self._end_time)):
                continue

            ts_and_files.append(TimestampFileObj(ts, file_object))

        # sort the incoming data by date
        ts_and_files.sort(key=lambda x: x.timestamp)

        # Create a dataframe with the incoming metadata
        if ((len(ts_and_files) > 1) and (self._sampling_rate_s > 0)):
            file_sampled_list = []

            ts_last = ts_and_files[0].timestamp

            file_sampled_list.append(ts_and_files[0])

            for idx in range(1, len(ts_and_files)):
                ts = ts_and_files[idx].timestamp

                if ((ts - ts_last).seconds >= self._sampling_rate_s):

                    ts_and_files.append(ts_and_files[idx])
                    ts_last = ts
            else:
                ts_and_files = file_sampled_list

        df = pd.DataFrame()

        timestamps = []
        full_names = []
        file_objs = []
        for (ts, file_object) in ts_and_files:
            timestamps.append(ts)
            full_names.append(file_object.full_name)
            file_objs.append(file_object)

        df["dfp_timestamp"] = timestamps
        df["key"] = full_names
        df["objects"] = file_objs

        output_batches = []

        if len(df) > 0:
            # Now split by the batching settings
            df_period = df["dfp_timestamp"].dt.to_period(self._period)

            period_gb = df.groupby(df_period)

            n_groups = len(period_gb)
            for group in period_gb.groups:
                period_df = period_gb.get_group(group)

                obj_list = fsspec.core.OpenFiles(period_df["objects"].to_list(),
                                                 mode=file_objects.mode,
                                                 fs=file_objects.fs)

                output_batches.append((obj_list, n_groups))

        return output_batches

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = builder.make_node(self.unique_name, ops.map(self.on_data), ops.flatten())
        builder.make_edge(input_stream[0], stream)

        return stream, typing.List[fsspec.core.OpenFiles]
