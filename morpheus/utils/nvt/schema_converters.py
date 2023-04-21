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


def sync_df_as_pandas(func: typing.Callable) -> typing.Callable:
    def wrapper(df: typing.Union[pd.DataFrame, cudf.DataFrame], **kwargs) -> typing.Union[pd.DataFrame, cudf.DataFrame]:
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

    json_flatten_op = [input_schema.json_columns] >> MutateOp(json_flatten, output_cols)

    return json_flatten_op


@sync_series_as_pandas
def datetime_converter(series: typing.Union[pd.Series, cudf.Series]) -> typing.Union[pd.Series, cudf.Series]:
    series = pd.to_datetime(series, infer_datetime_format=True, utc=True)

    return series


@sync_df_as_pandas
def string_cat_converter(df: typing.Union[pd.DataFrame, cudf.DataFrame], sep) -> typing.Union[
    pd.DataFrame, cudf.DataFrame]:
    return df.apply(lambda row: sep.join(row.values.astype(str)), axis=1)


def nvt_string_cat_converter(column_selector: ColumnSelector, df: typing.Union[pd.DataFrame, cudf.DataFrame],
                             sep: str = ', '):
    return string_cat_converter(df[column_selector.names], sep)


ColumnInfoProcessingMap = {
    BoolColumn: lambda ci: [ci.name] >> LambdaOp(lambda series: series.map(ci.value_map).astype(bool)),
    ColumnInfo: lambda ci: [ci.name] >> LambdaOp(lambda series: series),
    DateTimeColumn: lambda ci: [ci.name] >> LambdaOp(datetime_converter),
    RenameColumn: lambda ci: [ci.input_name] >> Rename(name=ci.name),
    StringCatColumn: lambda ci: [ci.input_columns] >> MutateOp(partial(nvt_string_cat_converter, sep=ci.sep), ci.name),
}


def input_schema_to_nvt_workflow(input_schema: DataFrameInputSchema) -> nvt.Workflow:
    """
    Converts an `input_schema` to a `nvt.Workflow` object

    First we aggregate all preprocessing steps, which we assume are independent of each other and can be run in parallel.

    Next we aggregate all column operations, which we assume are independent of each other and can be run in parallel and
    pass them the updated schema from the preprocessing steps.
    """

    # Items that need to be run before any other column operations, for example, we have to flatten json
    # columns so that we can do things like rename them.
    preprocess_workflow = None

    # Process the schema, so we can build up a list of workflow items
    if (input_schema.json_columns is not None and len(input_schema.json_columns) > 0):
        op = json_flatten_from_input_schema(input_schema)
        if (preprocess_workflow is None):
            preprocess_workflow = op
        else:
            preprocess_workflow = preprocess_workflow + op

    #for col_info in input_schema.column_info:
    #    if (type(col_info) not in ColumnInfoProcessingMap):
    #        raise RuntimeError(f"No known conversion for ColumnInfo type: {type(col_info)}")

    #    op = ColumnInfoProcessingMap[type(col_info)](col_info)
    #    if (preprocess_workflow is None):
    #        preprocess_workflow = op
    #    else:
    #        preprocess_workflow = preprocess_workflow >> op

    return nvt.Workflow(preprocess_workflow)
