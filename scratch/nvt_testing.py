import json
import typing
import pandas as pd

import cudf
import nvtabular as nvt

from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema, ColumnSchema
from nvtabular.ops.operator import ColumnSelector, Operator


class DynamicColumnSelector(ColumnSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resolve(self, schema: Schema) -> ColumnSelector:
        resolved_selector = super().resolve(schema)

        # Look for columns with the unique tag and add them to the selector
        print(f"\n ===========> Testing Schema Tags: {schema.column_names}", flush=True)
        tagged_columns = [col for col in schema.column_names if 'dynamically_created' in schema[col].tags]
        if tagged_columns:
            print(f"\n ===========> DETECTED DYNAMIC COLUMNS: {tagged_columns}")
            resolved_selector += ColumnSelector(names=tagged_columns)

        return resolved_selector


class MutateOp(Operator):
    def __init__(self, func, output_columns):
        super().__init__()

        self.func = func
        self.output_columns = output_columns or []

    @annotate("MutateOp", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        df = self.func(col_selector, df)
        print(f"\n =============> df: {df.columns}", flush=True)

        return df

    def compute_output_schema(
            self,
            input_schema: Schema,
            col_selector: ColumnSelector,
            prev_output_schema: typing.Optional[Schema] = None,
    ) -> Schema:
        output_schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        # Add new columns to the output schema
        for col, dtype in self.output_columns:
            output_schema += Schema([ColumnSchema(col, dtype=dtype, tags=["dynamically_created"])])

        return output_schema


class TestOutputOperator(Operator):
    def __init__(self):
        super().__init__()

    @annotate("TestOutputOperator", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    def compute_input_schema(
            self,
            root_schema: Schema,
            parents_schema: Schema,
            deps_schema: Schema,
            selector: ColumnSelector,
    ) -> Schema:
        print(f"=============> COMPUTING INPUT SCHEMA", flush=True)
        print(f"=============> parents_schema: {parents_schema.column_names}", flush=True)
        return super().compute_input_schema(root_schema, parents_schema, deps_schema, selector)


def json_flatten(column_selector, df):
    # Convert to pandas if it's a cudf DataFrame
    convert_to_cudf = False
    if isinstance(df, cudf.DataFrame):
        df = df.to_pandas()
        convert_to_cudf = True

    dtypes = df.dtypes
    print(f"Dtypes before: {dtypes}", flush=True)

    # Normalize JSON columns
    if (df["json_col1"].dtype == object):
        df_normalized = pd.json_normalize(df["json_col1"].apply(json.loads))
        df_normalized.reset_index(drop=True, inplace=True)
        df_normalized = pd.concat([df, df_normalized], axis=1)
        print("============> df_normalized: ", df_normalized.columns)
    else:
        # TODO(Devin) merlin doesn't support direct conversion to cudf StructDtype
        raise ValueError("JSON column must be of type str")

    # Convert back to cudf if necessary
    if convert_to_cudf:
        df_normalized = cudf.from_pandas(df_normalized)

    df_normalized["timestamp"] = df_normalized["timestamp"].astype("datetime64[us]")
    dtypes = df_normalized.dtypes
    print(f"Dtypes after: {dtypes}", flush=True)

    return df_normalized


def new_col_adder(column_selector, df):
    df["new_col1"] = 1
    df["new_col2"] = 1.5

    return df


pd.set_option("display.max_colwidth", None)
cdf = cudf.read_parquet("./test_json_input.parquet")

nvt_dataset = nvt.Dataset(cdf)
col_selector = ColumnSelector("*")
json_columns = [("id", "int64"), ("name", "str"), ("age", "int64"), ("email", "str"),
                ("address.street", "str"), ("address.city", "str"), ("address.state", "str"), ("address.zip", "str")]
output_columns = [("new_col1", "int64"), ("new_col2", "float64")]
test_op = col_selector >> MutateOp(json_flatten, json_columns) >> MutateOp(new_col_adder,
                                                                           output_columns) >> TestOutputOperator()

workflow = nvt.Workflow(test_op)
result = workflow.transform(nvt_dataset).to_ddf().compute()

print(f"\n =================> Returning result: {result.columns}", flush=True)
print(result.head(10))
