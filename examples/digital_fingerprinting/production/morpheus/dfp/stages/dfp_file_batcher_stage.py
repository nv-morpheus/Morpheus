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
import warnings
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
                 sampling_rate_s: typing.Optional[int] = None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 sampling: typing.Union[str, float, int, None] = None):
        super().__init__(c)

        self._date_conversion_func = date_conversion_func
        self._period = period
        self._start_time = start_time
        self._end_time = end_time

        if (sampling_rate_s is not None and sampling_rate_s > 0):
            assert sampling is None, "Cannot set both sampling and sampling_rate_s at the same time"

            # Show the deprecation message
            warnings.warn(("The `sampling_rate_s` argument has been deprecated. "
                           "Please use `sampling={sampling_rate_s}S` instead"),
                          DeprecationWarning)

            sampling = f"{sampling_rate_s}S"

        self._sampling = sampling

    @property
    def name(self) -> str:
        return "dfp-file-batcher"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (fsspec.core.OpenFiles, )

    def on_data(self, file_objects: fsspec.core.OpenFiles):

        timestamps = []
        full_names = []
        file_objs = []

        # Determine the date of the file, and apply the window filter if we have one
        ts_and_files = []
        for file_object in file_objects:
            ts = self._date_conversion_func(file_object)

            # Exclude any files outside the time window
            if ((self._start_time is not None and ts < self._start_time)
                    or (self._end_time is not None and ts > self._end_time)):
                continue

            timestamps.append(ts)
            full_names.append(file_object.full_name)
            file_objs.append(file_object)

        # Build the dataframe
        df = pd.DataFrame(index=pd.DatetimeIndex(timestamps), data={"filename": full_names, "objects": file_objects})

        # sort the incoming data by date
        df.sort_index(inplace=True)

        # If sampling was provided, perform that here
        if (self._sampling is not None):

            if (isinstance(self._sampling, str)):
                # We have a frequency for sampling. Resample by the frequency, taking the first
                df = df.resample(self._sampling).first().dropna()

            elif (self._sampling < 1.0):
                # Sample a fraction of the rows
                df = df.sample(frac=self._sampling).sort_index()

            else:
                # Sample a fixed amount
                df = df.sample(n=self._sampling).sort_index()

        # Early exit if no files were found
        if (len(df) == 0):
            return []

        if (self._period is None):
            # No period was set so group them all into one single batch
            return [(fsspec.core.OpenFiles(df["objects"].to_list(), mode=file_objects.mode, fs=file_objects.fs),
                     len(df))]

        # Now group the rows by the period
        resampled = df.resample(self._period)

        n_groups = len(resampled)

        output_batches = []

        for _, period_df in resampled:

            obj_list = fsspec.core.OpenFiles(period_df["objects"].to_list(), mode=file_objects.mode, fs=file_objects.fs)

            output_batches.append((obj_list, n_groups))

        return output_batches

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        stream = builder.make_node(self.unique_name, ops.map(self.on_data), ops.flatten())
        builder.make_edge(input_stream[0], stream)

        return stream, typing.List[fsspec.core.OpenFiles]
