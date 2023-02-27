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
import re
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import dateutil.parser
import fsspec
import fsspec.utils
import pandas as pd

import cudf

from morpheus.messages import MessageControl
from morpheus.messages.message_meta import MessageMeta
from morpheus.utils.file_utils import date_extractor
from morpheus.utils.loader_ids import FSSPEC_LOADER
from morpheus.utils.loader_utils import register_loader

logger = logging.getLogger(__name__)

dask_cluster = None


@register_loader(FSSPEC_LOADER)
def fsspec_loader(message: MessageControl, task: dict) -> MessageControl:

    files = task.get("files", [])
    start_time = task.get("start_time", None)
    duration = task.get("duration", None)
    sampling_rate_s = task.get("sampling_rate_s", 0)
    iso_date_regex_pattern = task.get("iso_date_regex_pattern", None)

    file_objects: fsspec.core.OpenFiles = fsspec.open_files(files)

    if (len(file_objects) == 0):
        raise RuntimeError(f"No files matched input strings: '{files}'. "
                           "Check your input pattern and ensure any credentials are correct")

    duration = timedelta(seconds=pd.Timedelta(duration).total_seconds())

    if start_time is None:
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - duration
    else:
        start_time = dateutil.parser.parse(start_time)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        end_time = start_time + duration

    TimestampFileObj = namedtuple("TimestampFileObj", ["timestamp", "file_name"])

    iso_date_regex = re.compile(iso_date_regex_pattern)

    ts_and_files = []

    for file_object in file_objects:
        ts = date_extractor(file_object, iso_date_regex)

        # Exclude any files outside the time window
        if ((start_time is not None and ts < start_time) or (end_time is not None and ts > end_time)):
            continue

        ts_and_files.append(TimestampFileObj(ts, file_object.full_name))

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

    df = cudf.DataFrame()

    timestamps = []
    full_names = []
    for (ts, file_name) in ts_and_files:
        timestamps.append(ts)
        full_names.append(file_name)

    df["ts"] = timestamps
    df["key"] = full_names

    message.payload(MessageMeta(df=df))

    return message
