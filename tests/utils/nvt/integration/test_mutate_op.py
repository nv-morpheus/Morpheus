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

import pandas as pd
import pytest
from merlin.dag import ColumnSelector

import cudf

from morpheus.utils.nvt.mutate import MutateOp
from morpheus.utils.nvt.transforms import json_flatten


@pytest.fixture(name="json_data")
def json_data_fixture():
    yield [
        '{"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}',
        '{"key1": "value2", "key2": {"subkey1": "subvalue3", "subkey2": "subvalue4"}}',
        '{"key1": "value3", "key2": {"subkey1": "subvalue5", "subkey2": "subvalue6"}}'
    ]


@pytest.fixture(name="expected_pdf")
def expected_pdf_fixture():
    yield pd.DataFrame({
        'col1.key1': ['value1', 'value2', 'value3'],
        'col1.key2.subkey1': ['subvalue1', 'subvalue3', 'subvalue5'],
        'col1.key2.subkey2': ['subvalue2', 'subvalue4', 'subvalue6']
    })


def test_integration_pandas(json_data: typing.List[str], expected_pdf: pd.DataFrame):
    pdf = pd.DataFrame({'col1': json_data})
    col_selector = ColumnSelector(['col1'])

    nvt_op = MutateOp(json_flatten, [("col1.key1", "object"), ("col1.key2.subkey1", "object"),
                                     ("col1.key2.subkey2", "object")])
    result_pdf = nvt_op.transform(col_selector, pdf)

    assert result_pdf.equals(expected_pdf), "Integration test with pandas DataFrame failed"


def test_integration_cudf(json_data: typing.List[str], expected_pdf: pd.DataFrame):
    cdf = cudf.DataFrame({'col1': json_data})
    col_selector = ColumnSelector(['col1'])

    nvt_op = MutateOp(json_flatten, [("col1.key1", "object"), ("col1.key2.subkey1", "object"),
                                     ("col1.key2.subkey2", "object")])
    result_cdf = nvt_op.transform(col_selector, cdf)
    result_pdf = result_cdf.to_pandas()

    assert result_pdf.equals(expected_pdf), "Integration test with cuDF DataFrame failed"
