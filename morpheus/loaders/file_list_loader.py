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

import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import time
import typing
from functools import partial

import fsspec
import fsspec.utils
from morpheus.messages.message_meta import MessageMeta

import pandas as pd

import cudf

from morpheus.messages import MessageControl
from morpheus._lib.common import FileTypes
from morpheus.cli.utils import str_to_file_type
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.column_info import process_dataframe
from morpheus.utils.loader_ids import FILE_LIST_LOADER
from morpheus.utils.loader_utils import register_loader

logger = logging.getLogger(__name__)

dask_cluster = None


@register_loader(FILE_LIST_LOADER)
def file_to_df_loader(message: MessageControl, task: dict):
    task_properties = task["properties"]
    files = task_properties["files"]

    file_objects: fsspec.core.OpenFiles = fsspec.open_files(files)

    if (len(file_objects) == 0):
        raise RuntimeError(f"No files matched input strings: '{files}'. "
                           "Check your input pattern and ensure any credentials are correct")

    files = None
    for file_object in file_objects:
        files.append(file_object.full_name)

    message_config = message.config()
    message_config["tasks"][0]["properties"]["files"] = files
    message_control = MessageControl(message_config)
    return message_control
