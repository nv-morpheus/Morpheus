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

import dataclasses
import logging
import re
import typing
from datetime import datetime

import nvtabular as nvt
import pandas as pd
import numpy as np

import cudf
from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema, ColumnSchema
from nvtabular.ops.operator import ColumnSelector, Operator

logger = logging.getLogger("morpheus.{}".format(__name__))


def create_increment_col(df, column_name: str, groupby_column="username", timestamp_column="timestamp"):
    """
    Create a new integer column counting unique occurrences of values in `column_name` grouped per-day using the
    timestamp values in `timestamp_column` and then grouping by `groupby_column` returning incrementing values starting
    at `1`.
    """
    DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'

    # Ensure we are pandas for this
    if (isinstance(df, cudf.DataFrame)):
        df = df.to_pandas()

    time_col = pd.to_datetime(df[timestamp_column], errors='coerce', utc=True).fillna(pd.to_datetime(DEFAULT_DATE))

    per_day = time_col.dt.to_period("D")

    cat_col: pd.Series = df.groupby([per_day, groupby_column
                                     ])[column_name].transform(lambda x: pd.factorize(x.fillna("nan"))[0] + 1)

    increment_col = pd.concat([cat_col, df[groupby_column]],
                              axis=1).groupby([per_day, groupby_column
                                               ])[column_name].expanding(1).max().droplevel(0).droplevel(0)

    return increment_col


def column_listjoin(df, col_name: str) -> pd.Series:
    """
    Returns the array series `df[col_name]` as flattened string series.
    """
    if col_name in df:
        return df[col_name].transform(lambda x: ",".join(x)).astype('string')
    else:
        return pd.Series(None, dtype='string')


@dataclasses.dataclass
class ColumnInfo:
    """Defines a single column and type-cast."""
    name: str
    dtype: str  # The final type

    def get_pandas_dtype(self) -> str:
        """Return the pandas type string of column."""
        if (issubclass(self.dtype, datetime)):
            return "datetime64[ns]"
        else:
            return self.dtype

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Performs the processing of the ColumnInfo. Most subclasses should override this method.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to process.

        Returns
        -------
        pd.Series
            New series defined by this class
        """
        if (self.name not in df.columns):
            return pd.Series(None, index=df.index, dtype=self.get_pandas_dtype())

        return df[self.name]


@dataclasses.dataclass
class CustomColumn(ColumnInfo):
    """Subclass of `ColumnInfo`, defines a column to be computed by a user-defined function `process_column_fn`."""
    process_column_fn: typing.Callable

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        return self.process_column_fn(df)


@dataclasses.dataclass
class RenameColumn(ColumnInfo):
    """Subclass of `ColumnInfo`, adds the ability to also perform a rename."""
    input_name: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        if (self.input_name not in df.columns):
            return pd.Series(None, index=df.index, dtype=self.get_pandas_dtype())

        return df[self.input_name]


@dataclasses.dataclass
class BoolColumn(RenameColumn):
    """Subclass of `RenameColumn`, adds the ability to map a set custom values as boolean values."""
    value_map: typing.Dict[str, bool] = dataclasses.field(init=False, default_factory=dict)

    true_value: dataclasses.InitVar[str] = None
    false_value: dataclasses.InitVar[str] = None

    true_values: dataclasses.InitVar[typing.List[str]] = None
    false_values: dataclasses.InitVar[typing.List[str]] = None

    def __post_init__(self,
                      true_value: str,
                      false_value: str,
                      true_values: typing.List[str],
                      false_values: typing.List[str]):
        if (true_value is not None):
            self.value_map.update({true_value: True})

        if (false_value is not None):
            self.value_map.update({false_value: False})

        if (true_values is not None):
            self.value_map.update({v: True for v in true_values})

        if (false_values is not None):
            self.value_map.update({v: False for v in false_values})

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        return super()._process_column(df).map(self.value_map).astype(bool)


@dataclasses.dataclass
class DateTimeColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, specific to casting UTC localized datetime values. When incoming values contain a
    time-zone offset string the values are converted to UTC, while values without a time-zone are assumed to be UTC.
    """

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(super()._process_column(df), infer_datetime_format=True, utc=True)


@dataclasses.dataclass
class StringJoinColumn(RenameColumn):
    """Subclass of `RenameColumn`, converts incoming `list` values to string by joining by `sep`."""
    sep: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        return super()._process_column(df).str.join(sep=self.sep)


@dataclasses.dataclass
class StringCatColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, concatenates values from multiple columns into a new string column separated by `sep`.
    """
    input_columns: typing.List[str]
    sep: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        first_col = df[self.input_columns[0]]

        return first_col.str.cat(others=df[self.input_columns[1:]], sep=self.sep)


@dataclasses.dataclass
class IncrementColumn(DateTimeColumn):
    """
    Subclass of `DateTimeColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time
    window `period` based on dates in the `input_name` field.
    """
    groupby_column: str
    period: str = "D"

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        period = super()._process_column(df).dt.to_period(self.period)

        # Create the `groupby_column`, per-period log count
        return df.groupby([self.groupby_column, period]).cumcount()


@dataclasses.dataclass
class DataFrameInputSchema:
    """Defines the schema specifying the columns to be included in the output `DataFrame`."""

    json_columns: typing.List[str] = dataclasses.field(default_factory=list)
    column_info: typing.List[ColumnInfo] = dataclasses.field(default_factory=list)
    preserve_columns: typing.List[str] = dataclasses.field(default_factory=list)
    row_filter: typing.Callable[[pd.DataFrame], pd.DataFrame] = None

    def __post_init__(self):

        input_preserve_columns = self.preserve_columns

        # Ensure preserve_columns is a list
        if (not isinstance(input_preserve_columns, list)):
            input_preserve_columns = [input_preserve_columns]

        # Compile the regex
        if (input_preserve_columns is not None and len(input_preserve_columns) > 0):
            input_preserve_columns = re.compile("({})".format("|".join(input_preserve_columns)))
        else:
            input_preserve_columns = None

        self.preserve_columns = input_preserve_columns


def _process_columns(df_in, input_schema: DataFrameInputSchema):
    # TODO(MDD): See what causes this to have such a perf impact over using df_in
    output_df = pd.DataFrame()

    convert_to_cudf = False

    if (isinstance(df_in, cudf.DataFrame)):
        df_in = df_in.to_pandas()
        convert_to_cudf = True

    # Iterate over the column info
    for ci in input_schema.column_info:
        try:
            output_df[ci.name] = ci._process_column(df_in)
        except Exception:
            logger.exception("Failed to process column '%s'. Dataframe: \n%s", ci.name, df_in, exc_info=True)
            raise

    if (input_schema.preserve_columns is not None):
        # Get the list of remaining columns not already added
        df_in_columns = set(df_in.columns) - set(output_df.columns)

        # Finally, keep any columns that match the preserve filters
        match_columns = [y for y in df_in_columns if input_schema.preserve_columns.match(y)]

        output_df[match_columns] = df_in[match_columns]

    if (convert_to_cudf):
        return cudf.from_pandas(output_df)

    print(f"\n_process_columns: {output_df.columns}", flush=True)
    return output_df


def _normalize_dataframe(df_in: pd.DataFrame, input_schema: DataFrameInputSchema):
    if (input_schema.json_columns is None or len(input_schema.json_columns) == 0):
        return df_in

    convert_to_cudf = False

    # Check if we are cudf
    if (isinstance(df_in, cudf.DataFrame)):
        df_in = df_in.to_pandas()
        convert_to_cudf = True

    json_normalized = []
    remaining_columns = list(df_in.columns)

    for j_column in input_schema.json_columns:

        if (j_column not in remaining_columns):
            continue

        normalized = pd.json_normalize(df_in[j_column])

        # Prefix the columns
        normalized.rename(columns={n: f"{j_column}.{n}" for n in normalized.columns}, inplace=True)

        # Reset the index otherwise there is a conflict
        normalized.reset_index(drop=True, inplace=True)

        json_normalized.append(normalized)

        # Remove from the list of remaining columns
        remaining_columns.remove(j_column)

    # Also need to reset the original index
    df_in.reset_index(drop=True, inplace=True)

    df_normalized = pd.concat([df_in[remaining_columns]] + json_normalized, axis=1)

    if (convert_to_cudf):
        return cudf.from_pandas(df_normalized)

    return df_normalized


def _filter_rows(df_in: pd.DataFrame, input_schema: DataFrameInputSchema):
    if (input_schema.row_filter is None):
        return df_in

    return input_schema.row_filter(df_in)


# TODO(Devin): Remove from inline after testing
class JSONNormalizeOp(Operator):
    def __init__(self, input_schema):
        super().__init__()
        self.input_schema = input_schema

    @annotate("JSONNormalizeOp", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        # Convert to pandas if it's a cudf DataFrame
        print(f"\n =============> APPLYING JSON NORMALIZE OP", flush=True)
        convert_to_cudf = False
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()
            convert_to_cudf = True

        # Normalize JSON columns
        df_normalized = _normalize_dataframe(df, self.input_schema)

        # Convert back to cudf if necessary
        if convert_to_cudf:
            df_normalized = cudf.from_pandas(df_normalized)

        print(f"\n =================> Returning df_normalized: {df_normalized.columns}", flush=True)
        return df_normalized


class TestOperator(Operator):
    def __init__(self):
        super().__init__()

    @annotate("TestOperator", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        df["new_col1"] = 0
        df["new_col2"] = 3.14
        print(f"\n =============> APPLYING TEST OPERATOR", flush=True)
        print(f"\n =============> df: {df.columns}", flush=True)

        return df

    # def compute_input_schema(
    #        self,
    #        root_schema: Schema,
    #        parents_schema: Schema,
    #        deps_schema: Schema,
    #        selector: ColumnSelector,
    # ) -> Schema:
    #    return super().compute_input_schema(root_schema, parents_schema, deps_schema, selector)

    def compute_output_schema(
            self,
            input_schema: Schema,
            col_selector: ColumnSelector,
            prev_output_schema: typing.Optional[Schema] = None,
    ) -> Schema:
        output_schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        # Add new columns to the output schema
        new_col1_schema = ColumnSchema("new_col1", dtype="int64", tags=["new"])
        new_col2_schema = ColumnSchema("new_col2", dtype="float64", tags=["new"])
        output_schema += Schema([new_col1_schema, new_col2_schema])

        return output_schema

    def compute_selector(
            self,
            input_schema: Schema,
            selector: ColumnSelector,
            parents_selector: typing.Optional[ColumnSelector] = None,
            dependencies_selector: typing.Optional[ColumnSelector] = None,
    ) -> ColumnSelector:
        selector = super().compute_selector(input_schema, selector, parents_selector, dependencies_selector)

        return selector

    def output_column_names(self, columns):
        columns_out = columns

        return columns_out

    # TODO(Devin): Write column mappings from schema json columns
    def column_mapping(self, col_selector):
        column_mapping = {}
        for col in col_selector.names:
            column_mapping[col] = [col]

        column_mapping["new_col1"] = [col]
        column_mapping["new_col2"] = [col]

        return column_mapping


def process_dataframe(df_in: pd.DataFrame, input_schema: DataFrameInputSchema) -> pd.DataFrame:
    """
    Applies column transformations as defined by `input_schema`
    """

    nvt_dataset = nvt.Dataset(df_in)
    cols = [x for x in df_in.columns]
    col_selector = ColumnSelector("*")
    json_normalize_op = col_selector >> JSONNormalizeOp(input_schema)
    test_op = col_selector >> TestOperator()

    workflow = nvt.Workflow(test_op + json_normalize_op)
    result = workflow.fit_transform(nvt_dataset).to_ddf().compute()

    print(f"\n =================> Returning result: {result.columns}", flush=True)

    # Step 1 is to normalize any columns
    df_processed = _normalize_dataframe(df_in, input_schema)

    # Step 2 is to process columns
    df_processed = _process_columns(df_processed, input_schema)

    # Step 3 is to run the row filter if needed
    df_processed = _filter_rows(df_processed, input_schema)

    return df_processed
