# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import functools
import os
import typing
from inspect import getsourcelines

from merlin.core.dispatch import DataFrameType
from merlin.schema import ColumnSchema
from merlin.schema import Schema
from nvtabular.ops.operator import ColumnSelector
from nvtabular.ops.operator import Operator

# Avoid using the annotate decorator in sphinx builds, instead define a simple pass-through decorator
if os.environ.get("MORPHEUS_IN_SPHINX_BUILD") is None:
    from merlin.core.dispatch import annotate  # pylint: disable=ungrouped-imports
else:

    def annotate(func, *args, **kwargs):  # pylint: disable=unused-argument

        @functools.wraps(func)
        def decorator(func):
            return func

        return decorator


class MutateOp(Operator):

    def __init__(self,
                 func: typing.Callable,
                 output_columns: typing.Optional[typing.List] = None,
                 dependencies: typing.Optional[typing.List] = None,
                 label: typing.Optional[str] = None):
        """
        Initialize MutateOp class.

        Parameters
        ----------
        func : Callable
            Function to perform mutation operation.
        output_columns : Optional[List], optional
            List of output columns, by default None.
        dependencies : Optional[List], optional
            List of dependencies, by default None.
        label : Optional[str], optional
            Label for MutateOp, by default None.
        """

        super().__init__()

        self._dependencies = dependencies or []
        self._func = func
        self._label = label
        self._output_columns = output_columns or []

    def _remove_deps(self, column_selector: ColumnSelector):
        """
        Remove dependencies from column selector.

        Parameters
        ----------
        column_selector : ColumnSelector
            Instance of ColumnSelector from which dependencies will be removed.

        Returns
        -------
        ColumnSelector
            Updated instance of ColumnSelector.
        """

        to_skip = ColumnSelector(
            [dep if isinstance(dep, str) else dep.output_schema.column_names for dep in self._dependencies])

        return column_selector.filter_columns(to_skip)

    @property
    def label(self):
        """
        Get the label of the MutateOp instance.

        Returns
        -------
        str
            The label of the MutateOp instance.
        """

        if (self._label is not None):
            return self._label

        # if we have a named function (not a lambda) return the function name
        name = self._func.__name__.split(".")[-1]
        if name != "<lambda>":
            return f"MutateOp: {name}"

        try:
            # otherwise get the lambda source code from the inspect module if possible
            source = getsourcelines(self.f)[0][0]
            lambdas = [op.strip() for op in source.split(">>") if "lambda " in op]
            if len(lambdas) == 1 and lambdas[0].count("lambda") == 1:
                return lambdas[0]
        except Exception:  # pylint: disable=broad-except
            # we can fail to load the source in distributed environments. Since the
            # label is mainly used for diagnostics, don't worry about the error here and
            # fallback to the default labelling
            pass

        # Failed to figure out the source
        return "MutateOp"

    # pylint: disable=arguments-renamed
    @annotate("MutateOp", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        """
        Apply the transformation function on the dataframe.

        Parameters
        ----------
        col_selector : ColumnSelector
            Instance of ColumnSelector.
        df : DataFrameType
            Input dataframe.

        Returns
        -------
        DataFrameType
            Transformed dataframe.
        """

        return self._func(col_selector, df)

    def column_mapping(self, col_selector: ColumnSelector) -> typing.Dict[str, str]:
        """
        Generate a column mapping.

        Parameters
        ----------
        col_selector : ColumnSelector
            Instance of ColumnSelector.

        Returns
        -------
        Dict[str, str]
            Dictionary of column mappings.
        """

        column_mapping = {}

        for col_name, _ in self._output_columns:
            column_mapping[col_name] = col_selector.names

        return column_mapping

    def compute_output_schema(
        self,
        input_schema: Schema,
        col_selector: ColumnSelector,
        prev_output_schema: typing.Optional[Schema] = None,
    ) -> Schema:
        """
        Compute the output schema.

        Parameters
        ----------
        input_schema : Schema
            The input schema.
        col_selector : ColumnSelector
            Instance of ColumnSelector.
        prev_output_schema : Optional[Schema], optional
            Previous output schema, by default None.

        Returns
        -------
        Schema
            The output schema.
        """
        output_schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        # Add new columns to the output schema
        for col, dtype in self._output_columns:
            output_schema += Schema([ColumnSchema(col, dtype=dtype)])

        return output_schema
