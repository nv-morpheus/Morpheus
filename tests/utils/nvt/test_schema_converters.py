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

import datetime
import pandas as pd

import nvtabular as nvt
from morpheus.utils.column_info import DataFrameInputSchema, ColumnInfo, RenameColumn, DateTimeColumn, StringCatColumn, \
    BoolColumn
from morpheus.utils.nvt.schema_converters import input_schema_to_nvt_workflow, json_flatten_from_input_schema

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


# Test 1: Test `json_flatten_from_input_schema` function
def test_json_flatten_from_input_schema():
    # Create a DataFrameInputSchema instance with the example schema provided
    example_schema = DataFrameInputSchema(
        json_columns=["access_device", "application", "auth_device", "user"],
        column_info=source_column_info
    )

    # Call `json_flatten_from_input_schema` with the created instance
    mutate_op = json_flatten_from_input_schema(example_schema)

    # Check if the returned `MutateOp` instance has the expected json_columns and output_columns
    assert mutate_op.op.label == "MutateOp"
    # assert set(mutate_op.output_columns) == set(
    #    [("user.name", str), ("access_device.browser", str), ("access_device.os", str), ("auth_device.name", str)])


# Test 2: Test `input_schema_to_nvt_workflow` function
def test_input_schema_to_nvt_workflow():
    # Create a DataFrameInputSchema instance with the example schema provided
    example_schema = DataFrameInputSchema(
        json_columns=["access_device", "application", "auth_device", "user"],
        column_info=source_column_info
    )

    # Call `input_schema_to_nvt_workflow` with the created instance
    workflow = input_schema_to_nvt_workflow(example_schema)

    # Check if the returned nvt.Workflow instance has the correct number of operations and if the operations are of the correct types
    # assert len(len(workflow.output_schema)) == 4
    assert set(workflow.output_schema) == set(["access_device", "application", "auth_device", "user"])


# Test 3: Test the conversion of a DataFrameInputSchema to an nvt.Workflow
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
    workflow = input_schema_to_nvt_workflow(example_schema)

    # Apply the returned nvt.Workflow to the test dataframe
    dataset = nvt.Dataset(test_df)
    output_df = workflow.transform(dataset).to_ddf().compute().to_pandas()

    # Check if the output dataframe has the expected schema and values
    expected_df = pd.DataFrame({
        "result": [True],
        "timestamp": [pd.Timestamp("2021-01-01 00:00:00")],
        "location": ["New York, NY, USA"],
        "authdevicename": ["Device1"],
        "username": ["John Doe"],
        "accessdevicebrowser": ["Chrome"],
        "accessdeviceos": ["Windows"],
    })

    pd.set_option('display.max_columns', None)
    # print("")
    # print(output_df)
    # print(output_df.columns)
    pd.testing.assert_frame_equal(output_df, expected_df)


if (__name__ in ('main',)):
    test_json_flatten_from_input_schema()
    test_input_schema_to_nvt_workflow()
    test_input_schema_conversion()
