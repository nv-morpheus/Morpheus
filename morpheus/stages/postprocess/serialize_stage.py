# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import copy
import re
import typing
from functools import partial

import mrc
from mrc.core import operators as ops

import morpheus._lib.stages as _stages
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MessageMeta
from morpheus.messages import MultiMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair


@register_stage("serialize")
class SerializeStage(SinglePortStage):
    """
    Includes & excludes columns from messages.

    This class filters columns from a `MultiMessage` object emitting a `MessageMeta`.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    include : typing.List[str], default = [], show_default="All Columns",
        Attributes that are required send to downstream stage.
    exclude : typing.List[str]
        Attributes that are not required send to downstream stage.
    fixed_columns : bool
        When `True` `SerializeStage` will assume that the Dataframe in all messages contain the same columns as the
        first message received.
    """

    def __init__(self,
                 c: Config,
                 include: typing.List[str] = [],
                 exclude: typing.List[str] = [r'^ID$', r'^_ts_'],
                 fixed_columns: bool = True):
        super().__init__(c)

        # Make copies of the arrays to prevent changes after the Regex is compiled
        self._include_columns = copy.copy(include)
        self._exclude_columns = copy.copy(exclude)
        self._fixed_columns = fixed_columns
        self._columns = None

    @property
    def name(self) -> str:
        return "serialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MultiMessage`, )
            Accepted input types.

        """
        return (MultiMessage, )

    def supports_cpp_node(self):
        # Enable support by default
        return True

    def convert_to_df(self,
                      x: MultiMessage,
                      include_columns: typing.Pattern,
                      exclude_columns: typing.List[typing.Pattern]):
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

        if self._fixed_columns and self._columns is not None:
            columns = self._columns
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

            self._columns = columns

        # Get metadata from columns
        df = x.get_meta(columns)

        return MessageMeta(df=df)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:
        if (self._build_cpp_node()):
            stream = _stages.SerializeStage(builder,
                                            self.unique_name,
                                            self._include_columns or [],
                                            self._exclude_columns,
                                            self._fixed_columns)
        else:
            include_columns = None

            if (self._include_columns is not None and len(self._include_columns) > 0):
                include_columns = re.compile("({})".format("|".join(self._include_columns)))

            exclude_columns = [re.compile(x) for x in self._exclude_columns]

            stream = builder.make_node(
                self.unique_name,
                ops.map(partial(self.convert_to_df, include_columns=include_columns, exclude_columns=exclude_columns)))

        builder.make_edge(input_stream[0], stream)

        return stream, MessageMeta
