#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from datetime import datetime
from functools import partial

import pandas as pd
import pytest

from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn
from morpheus.utils.column_info import process_dataframe
from utils import TEST_DIRS


@pytest.mark.use_python
def test_dataframe_input_schema_with_json_cols():

    src_file = os.path.join(TEST_DIRS.tests_data_dir, "azure_ad_logs.json")

    input_df = pd.read_json(src_file)

    raw_data_columns = [
        'time',
        'resourceId',
        'operationName',
        'operationVersion',
        'category',
        'tenantId',
        'resultType',
        'resultSignature',
        'resultDescription',
        'durationMs',
        'callerIpAddress',
        'correlationId',
        'identity',
        'Level',
        'location',
        'properties'
    ]

    assert len(input_df.columns) == 16
    assert list(input_df.columns) == raw_data_columns

    column_info = [
        DateTimeColumn(name="timestamp", dtype=datetime, input_name="time"),
        RenameColumn(name="userId", dtype=str, input_name="properties.userPrincipalName"),
        RenameColumn(name="appDisplayName", dtype=str, input_name="properties.appDisplayName"),
        ColumnInfo(name="category", dtype=str),
        RenameColumn(name="clientAppUsed", dtype=str, input_name="properties.clientAppUsed"),
        RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="properties.deviceDetail.browser"),
        RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="properties.deviceDetail.displayName"),
        RenameColumn(name="deviceDetailoperatingSystem",
                     dtype=str,
                     input_name="properties.deviceDetail.operatingSystem"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "properties.location.city",
                            "properties.location.countryOrRegion",
                        ],
                        sep=", "),
        RenameColumn(name="statusfailureReason", dtype=str, input_name="properties.status.failureReason"),
    ]

    schema = DataFrameInputSchema(json_columns=["properties"], column_info=column_info)

    df_processed = process_dataframe(input_df, schema)
    processed_df_cols = df_processed.columns

    assert len(input_df) == len(df_processed)
    assert len(processed_df_cols) == len(column_info)
    assert "timestamp" in processed_df_cols
    assert "userId" in processed_df_cols
    assert "time" not in processed_df_cols
    assert "properties.userPrincipalName" not in processed_df_cols


@pytest.mark.use_python
def test_dataframe_input_schema_without_json_cols():

    src_file = os.path.join(TEST_DIRS.tests_data_dir, "azure_ad_logs.json")

    input_df = pd.read_json(src_file)

    assert len(input_df.columns) == 16

    column_info = [
        DateTimeColumn(name="timestamp", dtype=datetime, input_name="time"),
        RenameColumn(name="userId", dtype=str, input_name="properties.userPrincipalName"),
        RenameColumn(name="appDisplayName", dtype=str, input_name="properties.appDisplayName"),
        ColumnInfo(name="category", dtype=str),
        RenameColumn(name="clientAppUsed", dtype=str, input_name="properties.clientAppUsed"),
        RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="properties.deviceDetail.browser"),
        RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="properties.deviceDetail.displayName"),
        RenameColumn(name="deviceDetailoperatingSystem",
                     dtype=str,
                     input_name="properties.deviceDetail.operatingSystem"),
        RenameColumn(name="statusfailureReason", dtype=str, input_name="properties.status.failureReason"),
    ]

    schema = DataFrameInputSchema(column_info=column_info)

    df_processed = process_dataframe(input_df, schema)
    processed_df_cols = df_processed.columns

    assert len(input_df) == len(df_processed)
    assert len(processed_df_cols) == len(column_info)
    assert "timestamp" in processed_df_cols
    assert "time" not in processed_df_cols
    assert "userId" in processed_df_cols
    assert len(df_processed[~df_processed.userId.isna()]) == 0

    column_info2 = [
        DateTimeColumn(name="timestamp", dtype=datetime, input_name="time"),
        RenameColumn(name="userId", dtype=str, input_name="properties.userPrincipalName"),
        RenameColumn(name="appDisplayName", dtype=str, input_name="properties.appDisplayName"),
        ColumnInfo(name="category", dtype=str),
        RenameColumn(name="clientAppUsed", dtype=str, input_name="properties.clientAppUsed"),
        RenameColumn(name="deviceDetailbrowser", dtype=str, input_name="properties.deviceDetail.browser"),
        RenameColumn(name="deviceDetaildisplayName", dtype=str, input_name="properties.deviceDetail.displayName"),
        RenameColumn(name="deviceDetailoperatingSystem",
                     dtype=str,
                     input_name="properties.deviceDetail.operatingSystem"),
        StringCatColumn(name="location",
                        dtype=str,
                        input_columns=[
                            "properties.location.city",
                            "properties.location.countryOrRegion",
                        ],
                        sep=", "),
        RenameColumn(name="statusfailureReason", dtype=str, input_name="properties.status.failureReason"),
    ]

    schema2 = DataFrameInputSchema(column_info=column_info2)

    # When trying to concat columns that don't exist in the dataframe, an exception is raised.
    with pytest.raises(Exception):
        process_dataframe(input_df, schema2)


@pytest.mark.use_python
def test_string_cat_column():

    cities = pd.Series([
        "New York",
        "Dallas",
        "Austin",
    ])

    countries = pd.Series([
        "USA",
        "USA",
        "USA",
    ])

    states = pd.Series([
        "New York",
        "Texas",
        "Texas",
    ])

    zipcodes = pd.Series([10001, 75001, 73301])

    df = pd.DataFrame({"city": cities, "country": countries, "state": states, "zipcode": zipcodes})

    string_cat_col = StringCatColumn(name="location", dtype=str, input_columns=[
        "city",
        "country",
    ], sep=", ")

    actual = string_cat_col.process_column(df)

    expected = pd.Series(["New York, USA", "Dallas, USA", "Austin, USA"])

    assert actual.equals(expected)

    string_cat_col_with_int = StringCatColumn(name="location", dtype=str, input_columns=[
        "city",
        "zipcode",
    ], sep=", ")

    with pytest.raises(Exception):
        string_cat_col_with_int.process_column(df)


@pytest.mark.use_python
def test_string_join_column():

    cities = pd.Series([
        "Boston",
        "Dallas",
        "Austin",
    ])

    df = pd.DataFrame({"city": cities})

    string_join_col = StringJoinColumn(name="city_new", dtype=str, input_name="city", sep="-")

    actual = string_join_col.process_column(df)

    expected = pd.Series(["B-o-s-t-o-n", "D-a-l-l-a-s", "A-u-s-t-i-n"])

    assert actual.equals(expected)


@pytest.mark.use_python
def test_column_info():

    cities = pd.Series([
        "Boston",
        "Dallas",
        "Austin",
    ])

    df = pd.DataFrame({"city": cities})

    string_join_col = ColumnInfo(name="city", dtype=str)

    actual = string_join_col.process_column(df)

    assert actual.equals(cities)
    assert string_join_col.name == "city"


@pytest.mark.use_python
def test_date_column():

    time_series = pd.Series([
        "2022-08-29T21:21:41.645157Z",
        "2022-08-29T21:23:19.500982Z",
        "2022-08-29T21:40:16.765798Z",
        "2022-08-29T22:23:15.895201Z",
        "2022-08-29T22:05:45.076460Z"
    ])

    df = pd.DataFrame({"time": time_series})

    datetime_col = DateTimeColumn(name="timestamp", dtype=datetime, input_name="time")

    datetime_series = datetime_col.process_column(df)

    assert datetime_series.dtype == 'datetime64[ns, UTC]'


@pytest.mark.use_python
def test_rename_column():

    time_series = pd.Series([
        "2022-08-29T21:21:41.645157Z",
        "2022-08-29T21:23:19.500982Z",
        "2022-08-29T21:40:16.765798Z",
        "2022-08-29T22:23:15.895201Z",
        "2022-08-29T22:05:45.076460Z"
    ])

    df = pd.DataFrame({"time": time_series})

    rename_col = RenameColumn(name="timestamp", dtype=str, input_name="time")

    actutal = rename_col.process_column(df)

    assert actutal.equals(time_series)


def convert_to_upper(df, column_name: str):
    return df[column_name].str.upper()


@pytest.mark.use_python
def test_custom_column():

    cities = pd.Series([
        "New York",
        "Dallas",
        "Austin",
    ])

    df = pd.DataFrame({"city": cities})

    custom_col = CustomColumn(name="city_upper",
                              dtype=str,
                              process_column_fn=partial(convert_to_upper, column_name="city"))

    actutal = custom_col.process_column(df)

    expected = pd.Series(["NEW YORK", "DALLAS", "AUSTIN"])

    assert actutal.equals(expected)
