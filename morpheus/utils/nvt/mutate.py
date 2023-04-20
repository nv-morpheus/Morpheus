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

from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema, ColumnSchema
from nvtabular.ops.operator import ColumnSelector, Operator

class MutateOp(Operator):
    def __init__(self, func, output_columns):
        super().__init__()

        self.func = func
        self.output_columns = output_columns or []

    @annotate("MutateOp", color="darkgreen", domain="nvt_python")
    def transform(self, column_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return self.func(column_selector, df)

    def column_mapping(self, col_selector: ColumnSelector) -> typing.Dict[str, str]:
        column_mapping = {}
        # TODO(Devin): Bit of a hack, but I don't want to pass a dictionary of all the source->dest mappings
        for col_name, _ in self.output_columns:
            column_mapping[col_name] = col_selector.names

        return column_mapping

    def compute_output_schema(
            self,
            input_schema: Schema,
            col_selector: ColumnSelector,
            prev_output_schema: typing.Optional[Schema] = None,
    ) -> Schema:
        output_schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        # Add new columns to the output schema
        for col, dtype in self.output_columns:
            output_schema += Schema([ColumnSchema(col, dtype=dtype, tags=["mutated"])])

        return output_schema
