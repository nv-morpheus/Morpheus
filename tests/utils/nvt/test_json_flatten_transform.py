# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import pandas as pd
import pytest
from nvtabular.ops.operator import ColumnSelector

import cudf

from morpheus.utils.nvt.transforms import json_flatten


@pytest.fixture(name="data")
def data_fixture():
    yield {
        "id": [1, 2],
        "info": [
            '{"name": "John", "age": 30, "city": "New York"}', '{"name": "Jane", "age": 28, "city": "San Francisco"}'
        ]
    }


def test_json_flatten_pandas(data: dict):
    df = pd.DataFrame(data)
    col_selector = ColumnSelector(["info"])
    result = json_flatten(col_selector, df)

    expected_data = {"info.name": ["John", "Jane"], "info.age": [30, 28], "info.city": ["New York", "San Francisco"]}
    expected_df = pd.DataFrame(expected_data)

    pd.testing.assert_frame_equal(result, expected_df)


def test_json_flatten_cudf(data: dict):
    df = cudf.DataFrame(data)
    col_selector = ColumnSelector(["info"])
    result = json_flatten(col_selector, df)

    expected_data = {
        "id": [1, 2], "info.name": ["John", "Jane"], "info.age": [30, 28], "info.city": ["New York", "San Francisco"]
    }
    expected_df = cudf.DataFrame(expected_data)

    assert_frame_equal(result, expected_df)


def assert_frame_equal(df1, df2):
    assert len(df1) == len(df2), "DataFrames have different lengths"
    for col in df1.columns:
        assert col in df2, f"Column {col} not found in the second DataFrame"
        assert (df1[col] == df2[col]).all(), f"Column {col} values do not match"
