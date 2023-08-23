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

import pandas as pd
import pytest
from nvtabular.ops.operator import ColumnSelector

from _utils.dataset_manager import DatasetManager
from morpheus.utils.nvt.transforms import json_flatten
from morpheus.utils.type_aliases import DataFrameType


@pytest.fixture(name="data")
def data_fixture():
    yield {
        "id": [1, 2],
        "info": [
            '{"name": "John", "age": 30, "city": "New York"}', '{"name": "Jane", "age": 28, "city": "San Francisco"}'
        ]
    }


@pytest.fixture(name="df")
def df_fixture(dataset: DatasetManager, data: dict):
    yield dataset.df_class(data)


def test_json_flatten(df: DataFrameType):
    col_selector = ColumnSelector(["info"])
    result = json_flatten(col_selector, df)

    expected_data = {"info.name": ["John", "Jane"], "info.age": [30, 28], "info.city": ["New York", "San Francisco"]}
    expected_df = pd.DataFrame(expected_data)

    DatasetManager.assert_df_equal(result, expected_df)
