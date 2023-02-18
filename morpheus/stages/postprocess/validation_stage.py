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
import json
import logging
import os
import typing

import mrc
import pandas as pd
from mrc.core import operators as ops

import cudf

from morpheus.common import FileTypes
from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.messages import MultiMessage
from morpheus.pipeline.multi_message_stage import MultiMessageStage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.utils import compare_df

logger = logging.getLogger(__name__)


@register_stage("validate")
class ValidationStage(MultiMessageStage):
    """
    Validate pipeline output for testing.

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
    overwrite : boolean, default = False, is_flag = True
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

    def supports_cpp_node(self):
        return False

    def _do_comparison(self, messages: typing.List[MultiMessage]):

        if (len(messages) == 0):
            return

        # Get all of the meta data and combine into a single frame
        all_meta = [x.get_meta() for x in messages]

        # Convert to pandas
        all_meta = [x.to_pandas() if isinstance(x, cudf.DataFrame) else x for x in all_meta]

        combined_df = pd.concat(all_meta)

        val_df = read_file_to_df(self._val_file_name, FileTypes.Auto, df_type="pandas")
        results = compare_df.compare_df(val_df,
                                        combined_df,
                                        self._include_columns,
                                        self._exclude_columns,
                                        replace_idx=self._index_col,
                                        abs_tol=self._abs_tol,
                                        rel_tol=self._rel_tol)

        with open(self._results_file_name, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        # Store all messages until on_complete is called and then build the dataframe and compare
        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):

            def do_compare(delayed_messages):

                self._do_comparison(delayed_messages)

                return delayed_messages

            obs.pipe(ops.to_list(), ops.map(do_compare), ops.flatten()).subscribe(sub)

        node = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
