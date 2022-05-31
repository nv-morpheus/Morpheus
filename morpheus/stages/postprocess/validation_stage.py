# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
import logging
import os
import re
import typing

import neo
import pandas as pd
from neo.core import operators as ops

import cudf

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair

logger = logging.getLogger(__name__)


class ValidationStage(MultiMessageStage):
    """
    The validation stage can be used to combine all output data into a single dataframe and compare against a known good
    file.

    Parameters
    ----------
    c : `morpheus.config.Config`
        The global configuration.
    val_file_name : str
        The comparison file.
    results_file_name : str
        Where to output a JSON containing the validation results.
    overwrite : bool, optional
        Whether to overwrite the validation results if they exist, by default False.
    include : typing.List[str], optional
        Any columns to include. By default all columns are included.
    exclude : typing.List[str], optional
        Any columns to exclude. Takes a regex, by default [r'^ID$', r'^_ts_'].
    index_col : str, optional
        Whether to convert any column in the dataset to the index. Useful when the pipeline will change the index,
        by default None.
    abs_tol : float, default = 0.001
        Absolute tolerance to use when comparing float columns.
    rel_tol : float, default = 0.05
        Relative tolerance to use when comparing float columns.

    Raises
    ------
    FileExistsError
        When overwrite is False and the results file exists.
    """

    def __init__(
        self,
        c: Config,
        val_file_name: str,
        results_file_name: str,
        overwrite: bool = False,
        include: typing.List[str] = None,
        exclude: typing.List[str] = [r'^ID$', r'^_ts_'],
        index_col: str = None,
        abs_tol: float = 0.001,
        rel_tol: float = 0.005,
    ):

        super().__init__(c)

        # Make copies of the arrays to prevent changes after the Regex is compiled
        self._include_columns = copy.copy(include)
        self._exclude_columns = copy.copy(exclude)
        self._index_col = index_col
        self._val_file_name = val_file_name
        self._results_file_name = results_file_name
        self._abs_tol = abs_tol
        self._rel_tol = rel_tol

        if (os.path.exists(self._results_file_name)):
            if (overwrite):
                os.remove(self._results_file_name)
            else:
                raise FileExistsError(
                    "Cannot output validation results to '{}'. File exists and overwrite = False".format(
                        self._results_file_name))

        self._val_df: pd.DataFrame = None

    @property
    def name(self) -> str:
        return "validation"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(`morpheus.pipeline.messages.MultiMessage`, )
            Accepted input types.

        """
        return (MultiMessage, )

    def _filter_df(self, df):
        include_columns = None

        if (self._include_columns is not None and len(self._include_columns) > 0):
            include_columns = re.compile("({})".format("|".join(self._include_columns)))

        exclude_columns = [re.compile(x) for x in self._exclude_columns]

        # Filter out any known good/bad columns we dont want to compare
        columns: typing.List[str] = []

        # First build up list of included. If no include regex is specified, select all
        if (include_columns is None):
            columns = list(df.columns)
        else:
            columns = [y for y in list(df.columns) if include_columns.match(y)]

        # Now remove by the ignore
        for test in exclude_columns:
            columns = [y for y in columns if not test.match(y)]

        return df[columns]

    def _do_comparison(self, messages: typing.List[MultiMessage]):

        if (len(messages) == 0):
            return

        import datacompy

        # Get all of the meta data and combine into a single frame
        all_meta = [x.get_meta() for x in messages]

        # Convert to pandas
        all_meta = [x.to_pandas() if isinstance(x, cudf.DataFrame) else x for x in all_meta]

        combined_df = pd.concat(all_meta)

        results_df = self._filter_df(combined_df)

        # if the index column is set, make that the index
        if (self._index_col is not None):
            results_df = results_df.set_index(self._index_col, drop=True)

            if (self._index_col.startswith("_index_")):
                results_df.index.name = str(results_df.index.name).replace("_index_", "", 1)

        val_df = self._filter_df(self._val_df)

        # Now start the comparison
        missing_columns = val_df.columns.difference(results_df.columns)
        extra_columns = results_df.columns.difference(val_df.columns)
        same_columns = val_df.columns.intersection(results_df.columns)

        # Now get the results in the same order
        results_df = results_df[same_columns]

        comparison = datacompy.Compare(
            val_df,
            results_df,
            on_index=True,
            abs_tol=self._abs_tol,
            rel_tol=self._rel_tol,
            df1_name="val",
            df2_name="res",
            cast_column_names_lower=False,
        )

        total_rows = len(val_df)
        diff_rows = len(val_df) - int(comparison.count_matching_rows())

        if (comparison.matches()):
            logger.info("Results match validation dataset")
        else:
            match_columns = comparison.intersect_rows[same_columns + "_match"]

            mismatched_idx = match_columns[match_columns.apply(lambda r: not r.all(), axis=1)].index

            merged = pd.concat([val_df, results_df], keys=["val", "res"]).swaplevel().sort_index()

            mismatch_df = merged.loc[mismatched_idx]

            logger.debug("Results do not match. Diff %d/%d (%f %%). First 10 mismatched rows:",
                         diff_rows,
                         total_rows,
                         diff_rows / total_rows * 100.0)
            logger.debug(mismatch_df[:20])

        # Now build the output
        output = {
            "total_rows": total_rows,
            "matching_rows": int(comparison.count_matching_rows()),
            "diff_rows": diff_rows,
            "matching_cols": list(same_columns),
            "extra_cols": list(extra_columns),
            "missing_cols": list(missing_columns),
        }

        with open(self._results_file_name, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

    def _build_single(self, seg: neo.Builder, input_stream: StreamPair) -> StreamPair:

        self._val_df: pd.DataFrame = read_file_to_df(self._val_file_name, FileTypes.Auto, df_type="pandas")

        # Store all messages until on_complete is called and then build the dataframe and compare
        def node_fn(input: neo.Observable, output: neo.Subscriber):

            def do_compare(delayed_messages):

                self._do_comparison(delayed_messages)

                return delayed_messages

            input.pipe(ops.to_list(), ops.map(do_compare), ops.flatten()).subscribe(output)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(input_stream[0], node)

        return node, input_stream[1]
