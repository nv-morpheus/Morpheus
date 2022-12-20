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
import re
from collections import namedtuple

import fsspec
import fsspec.utils
import mrc
import pandas as pd
from mrc.core import operators as ops

from morpheus.utils.file_utils import date_extractor
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(f"morpheus.{__name__}")


@register_module("FileBatcher", "morpheus_modules")
def file_batcher(builder: mrc.Builder):
    """
    This module loads the input files, removes files that are older than the chosen window of time,
    and then groups the remaining files by period that fall inside the window.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.
    """

    module_id = "FileBatcher"

    config = get_module_config(module_id, builder)

    TimestampFileObj = namedtuple("TimestampFileObj", ["timestamp", "file_object"])

    iso_date_regex_pattern = config.get("iso_date_regex_pattern", None)
    start_time = config.get("start_time", None)
    end_time = config.get("end_time", None)
    sampling_rate_s = config.get("sampling_rate_s", None)
    period = config.get("period", None)

    iso_date_regex = re.compile(iso_date_regex_pattern)

    def on_data(file_objects: fsspec.core.OpenFiles):

        # Determine the date of the file, and apply the window filter if we have one
        ts_and_files = []
        for file_object in file_objects:
            ts = date_extractor(file_object, iso_date_regex)

            # Exclude any files outside the time window
            if ((start_time is not None and ts < start_time) or (end_time is not None and ts > end_time)):
                continue

            ts_and_files.append(TimestampFileObj(ts, file_object))

        # sort the incoming data by date
        ts_and_files.sort(key=lambda x: x.timestamp)

        # Create a dataframe with the incoming metadata
        if ((len(ts_and_files) > 1) and (sampling_rate_s > 0)):
            file_sampled_list = []

            ts_last = ts_and_files[0].timestamp

            file_sampled_list.append(ts_and_files[0])

            for idx in range(1, len(ts_and_files)):
                ts = ts_and_files[idx].timestamp

                if ((ts - ts_last).seconds >= sampling_rate_s):

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

        df["ts"] = timestamps
        df["key"] = full_names
        df["objects"] = file_objs

        output_batches = []

        if len(df) > 0:
            # Now split by the batching settings
            df_period = df["ts"].dt.to_period(period)

            period_gb = df.groupby(df_period)

            n_groups = len(period_gb)
            for group in period_gb.groups:
                period_df = period_gb.get_group(group)

                obj_list = fsspec.core.OpenFiles(period_df["objects"].to_list(),
                                                 mode=file_objects.mode,
                                                 fs=file_objects.fs)

                output_batches.append((obj_list, n_groups))

        return output_batches

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
        obs.pipe(ops.map(on_data), ops.flatten()).subscribe(sub)

    node = builder.make_node_full(module_id, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
