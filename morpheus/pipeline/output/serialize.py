# Copyright (c) 2021, NVIDIA CORPORATION.
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
import json
import re
import typing

import cudf

from morpheus.config import Config
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair


class SerializeStage(SinglePortStage):
    """
    This class converts a `MultiMessage` object into a list of strings for writing out to file or Kafka.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    include : list[str]
        Attributes that are required send to downstream stage.
    exclude : typing.List[str]
        Attributes that are not required send to downstream stage.

    """
    def __init__(self,
                 c: Config,
                 include: typing.List[str] = None,
                 exclude: typing.List[str] = [r'^ID$', r'^ts_'],
                 as_cudf_df=False):
        super().__init__(c)

        # Make copies of the arrays to prevent changes after the Regex is compiled
        self._include_columns = copy.copy(include)
        self._exclude_columns = copy.copy(exclude)
        self._as_cudf_df = as_cudf_df

    @property
    def name(self) -> str:
        return "serialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(morpheus.pipeline.messages.MultiMessage, )
            Accepted input types

        """
        return (MultiMessage, )

    @staticmethod
    def convert_to_df(x: MultiMessage, include_columns: typing.Pattern, exclude_columns: typing.List[typing.Pattern]):
        """
        Converts dataframe to entries to JSON lines.

        Parameters
        ----------
        x : morpheus.pipeline.messages.MultiMessage
            MultiMessage instance that contains data.
        include_columns : typing.Pattern
            Columns that are required send to downstream stage.
        exclude_columns : typing.List[typing.Pattern]
            Columns that are not required send to downstream stage.

        """
        columns: typing.List[str] = []

        # First build up list of included. If no include regex is specified, select all
        if (include_columns is None):
            columns = list(x.meta.df.columns)
        else:
            columns = [y for y in list(x.meta.df.columns) if include_columns.match(y)]

        # Now remove by the ignore
        for test in exclude_columns:
            columns = [y for y in columns if not test.match(y)]

        # Get metadata from columns
        df = x.get_meta(columns)

        def double_serialize(y: str):
            try:
                return json.dumps(json.dumps(json.loads(y)))
            except:  # noqa: E722
                return y

        # Special processing for the data column (need to double serialize to match input)
        if ("data" in df):
            df["data"] = df["data"].apply(double_serialize)

        return df

    @staticmethod
    def convert_to_json(x: MultiMessage, include_columns: typing.Pattern, exclude_columns: typing.List[typing.Pattern]):

        df = SerializeStage.convert_to_df(x, include_columns=include_columns, exclude_columns=exclude_columns)

        # Convert to list of json string objects
        output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

        # Return list of strs to write out
        return output_strs

    @staticmethod
    def convert_to_cudf(x: MultiMessage, include_columns: typing.Pattern, exclude_columns: typing.List[typing.Pattern]):

        df = SerializeStage.convert_to_df(x, include_columns=include_columns, exclude_columns=exclude_columns)

        return cudf.from_pandas(df)

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        include_columns = None

        if (self._include_columns is not None and len(self._include_columns) > 0):
            include_columns = re.compile("({})".format("|".join(self._include_columns)))

        exclude_columns = [re.compile(x) for x in self._exclude_columns]

        # Convert the messages to rows of strings
        stream = input_stream[0].async_map(
            SerializeStage.convert_to_cudf if self._as_cudf_df else SerializeStage.convert_to_json,
            executor=self._pipeline.thread_pool,
            include_columns=include_columns,
            exclude_columns=exclude_columns)

        # Return input unchanged
        return stream, cudf.DataFrame if self._as_cudf_df else typing.List[str]
