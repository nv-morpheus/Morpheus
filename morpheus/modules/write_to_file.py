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
import os
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus._lib.common import FileTypes
from morpheus._lib.common import determine_file_type
from morpheus.io import serializers
from morpheus.messages.message_meta import MessageMeta
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import WRITE_TO_FILE
from morpheus.utils.module_utils import get_module_config
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)

is_first = True


@register_module(WRITE_TO_FILE, MORPHEUS_MODULE_NAMESPACE)
def write_to_file(builder: mrc.Builder):
    """
    Write all messages to a file.

    This module writes messages to a file.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.
    """

    config = builder.get_current_module_config()

    output_file = config.get("filename", None)
    overwrite = config.get("overwrite", False)
    flush = config.get("flush", False)
    file_type = config.get("file_type", FileTypes.Auto)
    include_index_col = config.get("include_index_col", True)

    if (os.path.exists(output_file)):
        if (overwrite):
            os.remove(output_file)
        else:
            raise FileExistsError(
                "Cannot output classifications to '{}'. File exists and overwrite = False".format(output_file))

    if (file_type == FileTypes.Auto):
        file_type = determine_file_type(output_file)

    def convert_to_strings(df: typing.Union[pd.DataFrame, cudf.DataFrame]):

        global is_first

        if (file_type == FileTypes.JSON):
            output_strs = serializers.df_to_json(df, include_index_col=include_index_col)
        elif (file_type == FileTypes.CSV):
            output_strs = serializers.df_to_csv(df, include_header=is_first, include_index_col=include_index_col)
        else:
            raise NotImplementedError("Unknown file type: {}".format(file_type))

        is_first = False

        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]

        return output_strs

    # Sink to file

    def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

        # Ensure our directory exists
        os.makedirs(os.path.realpath(os.path.dirname(output_file)), exist_ok=True)

        # Open up the file handle
        with open(output_file, "a") as out_file:
            def write_to_file(x: MessageMeta):
                lines = convert_to_strings(x.df)

                out_file.writelines(lines)

                if flush:
                    out_file.flush()

                return x

            obs.pipe(ops.map(write_to_file)).subscribe(sub)

        # File should be closed by here

    node = builder.make_node_full(WRITE_TO_FILE, node_fn)

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
