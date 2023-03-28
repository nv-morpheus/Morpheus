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
import typing
from functools import partial

import mrc
import pandas as pd

import cudf

from morpheus.messages import MultiMessage
from morpheus.messages.message_meta import MessageMeta
from morpheus.utils.module_ids import MORPHEUS_MODULE_NAMESPACE
from morpheus.utils.module_ids import SERIALIZE
from morpheus.utils.module_utils import register_module

logger = logging.getLogger(__name__)


@register_module(SERIALIZE, MORPHEUS_MODULE_NAMESPACE)
def serialize(builder: mrc.Builder):
    """
    Includes & excludes columns from messages.

    This module filters columns from a `MultiMessage` object emitting a `MessageMeta`.

    Parameters
    ----------
    builder : mrc.Builder
        mrc Builder object.

    Notes
    -----
        Configurable Parameters:
            - columns (list[string]): List of columns to include; Example: `["column1", "column2", "column3"]`;
            Default: None
            - exclude (list[string]): List of regex patterns to exclude columns; Example: `["column_to_exclude"]`;
            Default: `[r'^ID$', r'^_ts_']`
            - fixed_columns (bool): If true, the columns are fixed and not determined at runtime; Example: `true`;
            Default: true
            - include (string): Regex to include columns; Example: `^column`; Default: None
            - use_cpp (bool): If true, use C++ to serialize; Example: `true`; Default: false
    """

    config = builder.get_current_module_config()

    include_columns = config.get("include", None)
    exclude_columns = config.get("exclude", [r'^ID$', r'^_ts_'])
    fixed_columns = config.get("fixed_columns", True)
    columns = config.get("columns", None)
    use_cpp = config.get("use_cpp", False)

    def convert_to_df(x: MultiMessage,
                      include_columns: typing.Pattern,
                      exclude_columns: typing.List[typing.Pattern],
                      columns: typing.List[str]):
        """
        Converts dataframe to entries to JSON lines.

        Parameters
        ----------
        x : `morpheus.pipeline.messages.MultiMessage`
            MultiMessage instance that contains data.
        include_columns : typing.Pattern
            Columns that are required send to downstream stage.
        exclude_columns : typing.List[typing.Pattern]
            Columns that are not required send to downstream stage.

        """

        if fixed_columns and columns is not None:
            columns = columns
        else:
            columns: typing.List[str] = []

            # Minimize access to x.meta.df
            df_columns = list(x.meta.df.columns)

            # First build up list of included. If no include regex is specified, select all
            if (include_columns is None):
                columns = df_columns
            else:
                columns = [y for y in df_columns if include_columns.match(y)]

            # Now remove by the ignore
            for test in exclude_columns:
                columns = [y for y in columns if not test.match(y)]

            columns = columns

        # Get metadata from columns
        df = x.get_meta(columns)

        if (isinstance(df, pd.DataFrame) and use_cpp):
            df = cudf.from_pandas(df)

        return MessageMeta(df=df)

    if (include_columns is not None and len(include_columns) > 0):
        include_columns = re.compile("({})".format("|".join(include_columns)))

    exclude_columns = [re.compile(x) for x in exclude_columns]

    node = builder.make_node(
        SERIALIZE,
        partial(convert_to_df, include_columns=include_columns, exclude_columns=exclude_columns, columns=columns))

    # Register input and output port for a module.
    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
