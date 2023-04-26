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

import typing
from inspect import getsourcelines

from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema, ColumnSchema
from nvtabular.ops.operator import ColumnSelector, Operator


class MutateOp(Operator):
    def __init__(self, func, dependencies, output_columns):
        super().__init__()

        self._dependencies = dependencies
        self._func = func
        self._output_columns = output_columns or []

    def _remove_deps(self, column_selector):
        to_skip = ColumnSelector(
            [dep if isinstance(dep, str) else dep.output_schema.column_names for dep in self._dependencies]
        )

        return column_selector.filter_columns(to_skip)

    @property
    def label(self):
        # if we have a named function (not a lambda) return the function name
        name = self._func.__name__
        if name != "<lambda>":
            return f"MutateOp: {name}"

        else:
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

    @annotate("MutateOp", color="darkgreen", domain="nvt_python")
    def transform(self, column_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        column_selector = ColumnSelector(self._dependencies)
        _df = self._func(column_selector, df)

        return _df

    def column_mapping(self, col_selector: ColumnSelector, ) -> typing.Dict[str, str]:
        # filtered_selector = self._remove_deps(col_selector)
        # column_mapping = super().column_mapping(filtered_selector)
        column_mapping = {}

        for col_name, _ in self._output_columns:
            column_mapping[col_name] = []

        return column_mapping

    def compute_output_schema(
            self,
            input_schema: Schema,
            col_selector: ColumnSelector,
            prev_output_schema: typing.Optional[Schema] = None,
    ) -> Schema:
        output_schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        # Add new columns to the output schema
        for col, dtype in self._output_columns:
            output_schema += Schema([ColumnSchema(col, dtype=dtype, tags=["mutated"])])

        return output_schema
