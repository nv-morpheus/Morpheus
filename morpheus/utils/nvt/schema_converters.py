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

import cudf
import pandas as pd
import nvtabular as nvt
from functools import partial
from merlin.dag import ColumnSelector
from nvtabular.ops import LambdaOp, Rename

from morpheus.utils.column_info import (BoolColumn, ColumnInfo, DataFrameInputSchema, DateTimeColumn,
                                        RenameColumn, StringCatColumn)
from morpheus.utils.nvt import MutateOp, json_flatten
from nvtabular.workflow.node import WorkflowNode


def sync_df_as_pandas(func: typing.Callable) -> typing.Callable:
    def wrapper(df: typing.Union[pd.DataFrame, cudf.DataFrame], **kwargs) -> typing.Union[
        pd.DataFrame, cudf.DataFrame]:
        convert_to_cudf = False
        if type(df) == cudf.DataFrame:
            convert_to_cudf = True
            df = df.to_pandas()

        df = func(df, **kwargs)

        if convert_to_cudf:
            df = cudf.from_pandas(df)

        return df

    return wrapper


def sync_series_as_pandas(func: typing.Callable) -> typing.Callable:
    def wrapper(series: typing.Union[pd.Series, cudf.Series], **kwargs) -> typing.Union[pd.Series, cudf.Series]:
        convert_to_cudf = False
        if (type(series) == cudf.Series):
            convert_to_cudf = True
            series = series.to_pandas()

        series = func(series, **kwargs)

        if (convert_to_cudf):
            series = cudf.from_pandas(series)

        return series

    return wrapper


def json_flatten_from_input_schema(input_schema: DataFrameInputSchema) -> MutateOp:
    json_cols = set(input_schema.json_columns)

    json_output_candidates = []
    for col in input_schema.column_info:
        json_output_candidates.append((col.name, col.dtype))
        if (isinstance(col, StringCatColumn)):
            for col_name in col.input_columns:
                json_output_candidates.append((col_name, col.dtype))

    output_cols = []
    for col in json_output_candidates:
        cnsplit = col[0].split('.')
        if (len(cnsplit) > 1 and cnsplit[0] in json_cols):
            output_cols.append(col)

    print(json_cols)
    print([c[0] for c in json_output_candidates])
    print([c[0] for c in output_cols])

    if (output_cols is None or len(output_cols) == 0):
        raise RuntimeError("Failed to identify any output columns for json_flatten.")

    print(f"Creating with output cols: {output_cols}")
    jcols = [c for c in input_schema.json_columns]
    json_flatten_op = MutateOp(json_flatten, dependencies=jcols, output_columns=output_cols)

    return json_flatten_op


@sync_series_as_pandas
def datetime_converter(df: typing.Union[pd.DataFrame, cudf.DataFrame], output_column) -> typing.Union[
    pd.DataFrame, cudf.DataFrame]:
    _df = pd.DataFrame()
    _df[output_column] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, utc=True)

    return _df


def nvt_datetime_converter(column_selector: ColumnSelector, df: typing.Union[pd.DataFrame, cudf.DataFrame],
                           output_column: str) -> \
        typing.Union[pd.DataFrame, cudf.DataFrame]:
    return datetime_converter(df[column_selector.names], output_column=output_column)


@sync_df_as_pandas
def string_cat_col(df: typing.Union[pd.DataFrame, cudf.DataFrame], output_column, sep) -> typing.Union[
    pd.DataFrame, cudf.DataFrame]:
    cat_col = df.apply(lambda row: sep.join(row.values.astype(str)), axis=1)

    return pd.DataFrame({output_column: cat_col})


def nvt_string_cat_col(column_selector: ColumnSelector, df: typing.Union[pd.DataFrame, cudf.DataFrame],
                       output_column, input_columns, sep: str = ', '):
    return string_cat_col(df[input_columns], output_column=output_column, sep=sep)


ColumnInfoProcessingMap = {
    BoolColumn: lambda ci, deps: [ci.name] >> LambdaOp(lambda series: series.map(ci.value_map).astype(bool),
                                                       dtype="bool"),
    ColumnInfo: lambda ci, deps: LambdaOp(lambda series: series),
    DateTimeColumn: lambda ci, deps: [ci.input_name] >> Rename(name=ci.name) >>
                                     LambdaOp(lambda series: series.astype(ci.dtype), dtype=ci.dtype),
    RenameColumn: lambda ci, deps: [ci.input_name] >> Rename(name=ci.name),
    StringCatColumn: lambda ci, deps: MutateOp(
        partial(nvt_string_cat_col, output_column=ci.name, input_columns=ci.input_columns, sep=ci.sep),
        dependencies=deps, output_columns=[(ci.name, ci.dtype)]),
}


def coalesce_ops(ops: list):
    ops = [op for op in ops if op]  # Discard empty lists

    if len(ops) == 1:
        if isinstance(ops[0], list):
            return coalesce_ops(ops[0])
        else:
            return ops[0]

    operator_groups = []
    for op in ops:
        if isinstance(op, list):
            operator_groups.append(coalesce_ops(op))
        else:
            operator_groups.append(op)

    return operator_groups[0] + coalesce_ops(operator_groups[1:])


def input_schema_to_nvt_workflow(input_schema: DataFrameInputSchema) -> nvt.Workflow:
    """
    Converts an `input_schema` to a `nvt.Workflow` object

    First we aggregate all preprocessing steps, which we assume are independent of each other and can be run in parallel.

    Next we aggregate all column operations, which we assume are independent of each other and can be run in parallel and
    pass them the updated schema from the preprocessing steps.
    """

    # Items that need to be run before any other column operations, for example, we have to flatten json
    # columns so that we can do things like rename them.
    preprocess_workflow = []
    column_workflow = []

    # Process the schema, so we can build up a list of workflow items
    if (input_schema.json_columns is not None and len(input_schema.json_columns) > 0):
        op = ColumnSelector(input_schema.json_columns) >> json_flatten_from_input_schema(input_schema)
        preprocess_workflow.append(op)

    for col_info in input_schema.column_info:
        if (type(col_info) not in ColumnInfoProcessingMap):
            raise RuntimeError(f"No known conversion for ColumnInfo type: {type(col_info)}")

        op = ColumnInfoProcessingMap[type(col_info)](col_info, deps=[])
        column_workflow.append(op)

    preproc_workflow = coalesce_ops([preprocess_workflow])
    preproc_workflow = (preproc_workflow + ColumnSelector('*'))

    column_workflow_nodes = []
    for op in column_workflow:
        if (isinstance(op, WorkflowNode)):
            column_workflow_nodes.append(op)
        else:
            column_workflow_nodes.append(preproc_workflow >> op)

    compound_workflow = None
    for node in column_workflow_nodes:
        if (compound_workflow is None):
            compound_workflow = node
        else:
            compound_workflow = compound_workflow + node

    compound_workflow.graph.render(view=True, format='svg')

    return nvt.Workflow(compound_workflow)
