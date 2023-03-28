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

import json
import logging
import os
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import MultiMessage
from morpheus.pipeline.stream_pair import StreamPair
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage

logger = logging.getLogger(__name__)


@register_stage("validate")
class ValidationStage(CompareDataFrameStage):
    """
    Validate pipeline output for testing.

    The validation stage can be used to combine all output data into a single dataframe and compare against a known good
    file.

    If a column name is matched by both `include` and `exclude`, it will be excluded.

    Parameters
    ----------
    c : `morpheus.config.Config`
        The global configuration.
    val_file_name : str
        The comparison file, or an instance of a DataFrame.
    results_file_name : str, optional
        If not `None` specifies an output file path to write a JSON file containing the validation results.
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
        results_file_name: str = None,
        overwrite: bool = False,
        include: typing.List[str] = None,
        exclude: typing.List[str] = [r'^ID$', r'^_ts_'],
        index_col: str = None,
        abs_tol: float = 0.001,
        rel_tol: float = 0.005,
    ):

        super().__init__(c,
                         compare_df=val_file_name,
                         include=include,
                         exclude=exclude,
                         index_col=index_col,
                         abs_tol=abs_tol,
                         rel_tol=rel_tol)

        self._results_file_name = results_file_name

        if (self._results_file_name is not None and os.path.exists(self._results_file_name)):
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

    def _do_comparison(self):
        results = self.get_results(clear=False)
        if (len(results) and self._results_file_name is not None):
            with open(self._results_file_name, "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)

    def _build_single(self, builder: mrc.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: mrc.Observable, sub: mrc.Subscriber):
            obs.pipe(ops.map(self._append_message), ops.on_completed(self._do_comparison)).subscribe(sub)

        node = builder.make_node(self.unique_name, ops.build(node_fn))
        builder.make_edge(input_stream[0], node)

        return node, input_stream[1]
