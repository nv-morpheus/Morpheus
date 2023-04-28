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

import cudf
import pytest

import networkx as nx
import nvtabular as nvt
import pandas as pd

from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn

from morpheus.utils.nvt.schema_converters import JSONFlattenInfo
from morpheus.utils.nvt.schema_converters import input_schema_to_nvt_workflow
from morpheus.utils.nvt.schema_converters import get_ci_column_selector
from morpheus.utils.nvt.schema_converters import resolve_json_output_columns
from morpheus.utils.nvt.schema_converters import sync_df_as_pandas
from morpheus.utils.nvt.schema_converters import coalesce_ops
from morpheus.utils.nvt.schema_converters import json_flatten_from_input_schema

source_column_info = [
    BoolColumn(name="result",
               dtype="bool",
               input_name="result",
               true_values=["success", "SUCCESS"],
               false_values=["denied", "DENIED", "FRAUD"]),
    ColumnInfo(name="reason", dtype=str),
    DateTimeColumn(name="timestamp", dtype="datetime64[us]", input_name="timestamp"),
    StringCatColumn(name="location",
                    dtype="str",
                    input_columns=[
                        "access_device.location.city",
                        "access_device.location.state",
                        "access_device.location.country"
                    ], sep=", "),
    RenameColumn(name="authdevicename", dtype="str", input_name="auth_device.name"),
    RenameColumn(name="username", dtype="str", input_name="user.name"),
    RenameColumn(name="accessdevicebrowser", dtype="str", input_name="access_device.browser"),
    RenameColumn(name="accessdeviceos", dtype="str", input_name="access_device.os"),
]


def create_test_dataframe():
    return pd.DataFrame({
        "access_device": [
            '{"browser": "Firefox", "os": "Linux", "location": {"city": "San Francisco", "state": "CA", "country": "USA"}}'
        ],
        "user.name": ["Jane Smith"],
        "application": ['{"name": "AnotherApp"}'],
        "auth_device": ['{"name": "Device2"}'],
        "user": ['{"name": "Jane Smith"}'],
        "timestamp": [pd.Timestamp("2021-02-02 12:00:00")],
        "result": ["denied"],
        "reason": ["Denied"]
    })


def test_sync_df_as_pandas_pd_dataframe():
    @sync_df_as_pandas
    def test_func(df: pd.DataFrame, value: int) -> pd.DataFrame:
        df['test_col'] = df['test_col'] * value
        return df

    df = pd.DataFrame({'test_col': [1, 2, 3]})
    result = test_func(df, value=2)
    expected = pd.DataFrame({'test_col': [2, 4, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_sync_df_as_pandas_cudf_dataframe():
    @sync_df_as_pandas
    def test_func(df: pd.DataFrame, value: int) -> pd.DataFrame:
        df['test_col'] = df['test_col'] * value
        return df

    df = cudf.DataFrame({'test_col': [1, 2, 3]})
    result = test_func(df, value=2)
    expected = cudf.DataFrame({'test_col': [2, 4, 6]})
    cudf.testing.assert_frame_equal(result, expected)


def test_json_flatten_info_init():
    ci = JSONFlattenInfo(name="json_info", dtype="str",
                         input_col_names=["json_col1.a", "json_col2.b"],
                         output_col_names=["json_output_col1", "json_output_col2"])
    assert ci.name == "json_info"
    assert ci.dtype == "str"
    assert ci.input_col_names == ["json_col1.a", "json_col2.b"]
    assert ci.output_col_names == ["json_output_col1", "json_output_col2"]


def test_json_flatten_info_init_missing_input_col_names():
    with pytest.raises(TypeError):
        ci = JSONFlattenInfo(name="json_info", dtype="str",
                             output_col_names=["json_output_col1", "json_output_col2"])


def test_json_flatten_info_init_missing_output_col_names():
    with pytest.raises(TypeError):
        ci = JSONFlattenInfo(name="json_info", dtype="str",
                             input_col_names=["json_col1.a", "json_col2.b"])


def test_get_ci_column_selector_rename_column():
    ci = RenameColumn(input_name="original_name", name="new_name", dtype="str")
    result = get_ci_column_selector(ci)
    assert result == "original_name"


def test_get_ci_column_selector_bool_column():
    ci = BoolColumn(input_name="original_name", name="new_name", dtype="bool", true_values=["True"],
                    false_values=["False"])
    result = get_ci_column_selector(ci)
    assert result == "original_name"


def test_get_ci_column_selector_datetime_column():
    ci = DateTimeColumn(input_name="original_name", name="new_name", dtype="datetime64[ns]")
    result = get_ci_column_selector(ci)
    assert result == "original_name"


def test_get_ci_column_selector_string_join_column():
    ci = StringJoinColumn(input_name="original_name", name="new_name", dtype="str", sep=",")
    result = get_ci_column_selector(ci)
    assert result == "original_name"


def test_get_ci_column_selector_increment_column():
    ci = IncrementColumn(input_name="original_name", name="new_name", dtype="datetime64[ns]",
                         groupby_column="groupby_col")
    result = get_ci_column_selector(ci)
    assert result == "original_name"


def test_get_ci_column_selector_string_cat_column():
    ci = StringCatColumn(name="new_name", dtype="str", input_columns=["col1", "col2"], sep=", ")
    result = get_ci_column_selector(ci)
    assert result == ["col1", "col2"]


def test_get_ci_column_selector_json_flatten_info():
    ci = JSONFlattenInfo(name="json_info", dtype="str", input_col_names=["json_col1.a", "json_col2.b"],
                         output_col_names=["json_col1_a", "json_col2_b"])
    result = get_ci_column_selector(ci)
    assert result == ["json_col1.a", "json_col2.b"]


def test_resolve_json_output_columns():
    input_schema = DataFrameInputSchema(
        json_columns=["json_col"],
        column_info=[
            BoolColumn(input_name="bool_col", name="bool_col", dtype="bool", true_values=["True"],
                       false_values=["False"]),
            DateTimeColumn(input_name="datetime_col", name="datetime_col", dtype="datetime64[ns]"),
            RenameColumn(input_name="json_col.a", name="new_rename_col", dtype="str"),
            StringCatColumn(name="new_str_cat_col", dtype="str", input_columns=["A", "B"], sep=", "),
        ]
    )

    output_cols = resolve_json_output_columns(input_schema)
    expected_output_cols = [
        ("json_col.a", "str"),
    ]
    assert output_cols == expected_output_cols


def test_resolve_json_output_columns_empty_input_schema():
    input_schema = DataFrameInputSchema()
    output_cols = resolve_json_output_columns(input_schema)
    assert output_cols == []


def test_resolve_json_output_columns_no_json_columns():
    input_schema = DataFrameInputSchema(
        column_info=[
            ColumnInfo(name="column1", dtype="int"),
            ColumnInfo(name="column2", dtype="str")
        ]
    )
    output_cols = resolve_json_output_columns(input_schema)
    assert output_cols == []


def test_resolve_json_output_columns_with_json_columns():
    input_schema = DataFrameInputSchema(
        json_columns=["json_col"],
        column_info=[
            ColumnInfo(name="json_col.a", dtype="str"),
            ColumnInfo(name="json_col.b", dtype="int"),
            ColumnInfo(name="column3", dtype="float")
        ]
    )
    output_cols = resolve_json_output_columns(input_schema)
    assert output_cols == [("json_col.a", "str"), ("json_col.b", "int")]


def test_resolve_json_output_columns_with_complex_schema():
    input_schema = DataFrameInputSchema(
        json_columns=["json_col"],
        column_info=[
            ColumnInfo(name="json_col.a", dtype="str"),
            ColumnInfo(name="json_col.b", dtype="int"),
            ColumnInfo(name="column3", dtype="float"),
            RenameColumn(name="new_column", dtype="str", input_name="column4")
        ]
    )
    output_cols = resolve_json_output_columns(input_schema)
    assert output_cols == [("json_col.a", "str"), ("json_col.b", "int")]


def test_input_schema_conversion_empty_schema():
    empty_schema = DataFrameInputSchema()

    with pytest.raises(ValueError, match="Input schema is empty"):
        workflow = input_schema_to_nvt_workflow(empty_schema)  # noqa


def test_input_schema_conversion_additional_column():
    additional_column = RenameColumn(name="appname", dtype="str", input_name="application.name")
    modified_source_column_info = source_column_info + [additional_column]

    modified_schema = DataFrameInputSchema(
        json_columns=["access_device", "application", "auth_device", "user"],
        column_info=modified_source_column_info
    )
    test_df = create_test_dataframe()

    workflow = input_schema_to_nvt_workflow(modified_schema)
    dataset = nvt.Dataset(test_df)
    output_df = workflow.transform(dataset).to_ddf().compute().to_pandas()

    expected_df = pd.DataFrame({
        "result": [False],
        "reason": ["Denied"],
        "timestamp": [pd.Timestamp("2021-02-02 12:00:00")],
        "location": ["San Francisco, CA, USA"],
        "authdevicename": ["Device2"],
        "username": ["Jane Smith"],
        "accessdevicebrowser": ["Firefox"],
        "accessdeviceos": ["Linux"],
        "appname": ["AnotherApp"]
    })

    pd.testing.assert_frame_equal(output_df, expected_df)


def test_input_schema_conversion_interdependent_columns():
    additional_column_1 = StringCatColumn(name="fullname", dtype="str",
                                          input_columns=["user.firstname", "user.lastname"], sep=" ")
    additional_column_2 = StringCatColumn(name="appinfo", dtype="str",
                                          input_columns=["application.name", "application.version"], sep="-")

    modified_source_column_info = source_column_info + [additional_column_1, additional_column_2]

    modified_schema = DataFrameInputSchema(
        json_columns=["access_device", "application", "auth_device", "user"],
        column_info=modified_source_column_info
    )

    test_df = create_test_dataframe()
    test_df["user"] = ['{"firstname": "Jane", "lastname": "Smith", "name": "Jane Smith"}']
    test_df["application"] = ['{"name": "AnotherApp", "version": "1.0"}']

    workflow = input_schema_to_nvt_workflow(modified_schema)
    dataset = nvt.Dataset(test_df)
    output_df = workflow.transform(dataset).to_ddf().compute().to_pandas()

    expected_df = pd.DataFrame({
        "result": [False],
        "reason": ["Denied"],
        "timestamp": [pd.Timestamp("2021-02-02 12:00:00")],
        "location": ["San Francisco, CA, USA"],
        "authdevicename": ["Device2"],
        "username": ["Jane Smith"],
        "accessdevicebrowser": ["Firefox"],
        "accessdeviceos": ["Linux"],
        "fullname": ["Jane Smith"],
        "appinfo": ["AnotherApp-1.0"]
    })

    pd.testing.assert_frame_equal(output_df, expected_df)


def test_input_schema_conversion_nested_operations():
    additional_column_1 = StringCatColumn(name="rootcat", dtype="str", input_columns=["lhs_top_level", "rhs_top_level"],
                                          sep="-")
    additional_column_2 = RenameColumn(name="rhs_top_level", dtype="str", input_name="rhs_top_level_pre")
    additional_column_3 = ColumnInfo(name="lhs_top_level", dtype="str")
    modified_source_column_info = [additional_column_1, additional_column_2, additional_column_3]

    modified_schema = DataFrameInputSchema(
        json_columns=[],
        column_info=modified_source_column_info
    )

    test_df = create_test_dataframe()
    test_df["lhs_top_level"] = ["lhs"]
    test_df["rhs_top_level_pre"] = ["rhs"]

    workflow = input_schema_to_nvt_workflow(modified_schema, visualize=True)
    dataset = nvt.Dataset(test_df)
    output_df = workflow.transform(dataset).to_ddf().compute().to_pandas()

    expected_df = pd.DataFrame({
        "rootcat": ["lhs-rhs"]
    })

    pd.testing.assert_frame_equal(output_df, expected_df)


# Test the conversion of a DataFrameInputSchema to an nvt.Workflow
def test_input_schema_conversion():
    # Create a DataFrameInputSchema instance with the example schema provided
    example_schema = DataFrameInputSchema(
        json_columns=["access_device", "application", "auth_device", "user"],
        column_info=source_column_info
    )

    # Create a test dataframe with data according to the schema
    test_df = pd.DataFrame({
        "access_device": [
            '{"browser": "Chrome", "os": "Windows", "location": {"city": "New York", "state": "NY", "country": "USA"}}'],
        "user.name": ["John Doe"],
        "application": ['{"name": "TestApp"}'],
        "auth_device": ['{"name": "Device1"}'],
        "user": ['{"name": "John Doe"}'],
        "timestamp": [pd.Timestamp("2021-01-01 00:00:00")],
        "result": ["SUCCESS"],
        "reason": ["Authorized"]
    })

    # Call `input_schema_to_nvt_workflow` with the created instance
    workflow = input_schema_to_nvt_workflow(example_schema, visualize=True)

    # Apply the returned nvt.Workflow to the test dataframe
    dataset = nvt.Dataset(test_df)
    output_df = workflow.transform(dataset).to_ddf().compute().to_pandas()

    # Check if the output dataframe has the expected schema and values
    expected_df = pd.DataFrame({
        "result": [True],
        "reason": ["Authorized"],
        "timestamp": [pd.Timestamp("2021-01-01 00:00:00")],
        "location": ["New York, NY, USA"],
        "authdevicename": ["Device1"],
        "username": ["John Doe"],
        "accessdevicebrowser": ["Chrome"],
        "accessdeviceos": ["Windows"],
    })

    pd.set_option('display.max_columns', None)
    pd.testing.assert_frame_equal(output_df, expected_df)


if (__name__ in ('main',)):
    test_input_schema_conversion()
