# Copyright (c) 2023, NVIDIA CORPORATION.
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

from dataclasses import dataclass
from datetime import datetime
from functools import partial

from morpheus.config import Config
from morpheus.utils.column_info import BoolColumn
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import CustomColumn
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import create_increment_col


@dataclass
class Schema:
    source: DataFrameInputSchema
    preprocess: DataFrameInputSchema


class SchemaBuilder:

    def __init__(self, config: Config, log_type: str):
        self._config = config
        self._log_type = log_type

    def build_schema(self):

        if self._log_type == "duo":
            return self._build_duo_schema()
        elif self._log_type == "azure":
            return self._build_azure_schema()
        else:
            raise Exception("No matching schema found for log type : {}".format(self._log_type))

    def _build_azure_schema(self) -> Schema:
        # Specify the column names to ensure all data is uniform
        source_column_info = [
            DateTimeColumn(name=self._config.ae.timestamp_column_name, dtype=datetime, input_name="time"),
            RenameColumn(name=self._config.ae.userid_column_name, dtype=str, input_name="properties.userPrincipalName"),
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

        source_schema = DataFrameInputSchema(json_columns=["properties"], column_info=source_column_info)

        preprocess_column_info = [
            ColumnInfo(name=self._config.ae.timestamp_column_name, dtype=datetime),
            ColumnInfo(name=self._config.ae.userid_column_name, dtype=str),
            ColumnInfo(name="appDisplayName", dtype=str),
            ColumnInfo(name="clientAppUsed", dtype=str),
            ColumnInfo(name="deviceDetailbrowser", dtype=str),
            ColumnInfo(name="deviceDetaildisplayName", dtype=str),
            ColumnInfo(name="deviceDetailoperatingSystem", dtype=str),
            ColumnInfo(name="statusfailureReason", dtype=str),

            # Derived columns
            IncrementColumn(name="logcount",
                            dtype=int,
                            input_name=self._config.ae.timestamp_column_name,
                            groupby_column=self._config.ae.userid_column_name),
            CustomColumn(name="locincrement",
                         dtype=int,
                         process_column_fn=partial(create_increment_col, column_name="location")),
            CustomColumn(name="appincrement",
                         dtype=int,
                         process_column_fn=partial(create_increment_col, column_name="appDisplayName")),
        ]

        preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

        schema = Schema(source=source_schema, preprocess=preprocess_schema)

        return schema

    def _build_duo_schema(self) -> Schema:

        # Specify the column names to ensure all data is uniform
        source_column_info = [
            DateTimeColumn(name=self._config.ae.timestamp_column_name, dtype=datetime, input_name="timestamp"),
            RenameColumn(name=self._config.ae.userid_column_name, dtype=str, input_name="user.name"),
            RenameColumn(name="accessdevicebrowser", dtype=str, input_name="access_device.browser"),
            RenameColumn(name="accessdeviceos", dtype=str, input_name="access_device.os"),
            StringCatColumn(name="location",
                            dtype=str,
                            input_columns=[
                                "access_device.location.city",
                                "access_device.location.state",
                                "access_device.location.country"
                            ],
                            sep=", "),
            RenameColumn(name="authdevicename", dtype=str, input_name="auth_device.name"),
            BoolColumn(name="result",
                       dtype=bool,
                       input_name="result",
                       true_values=["success", "SUCCESS"],
                       false_values=["denied", "DENIED", "FRAUD"]),
            ColumnInfo(name="reason", dtype=str),
        ]

        source_schema = DataFrameInputSchema(json_columns=["access_device", "application", "auth_device", "user"],
                                             column_info=source_column_info)

        # Preprocessing schema
        preprocess_column_info = [
            ColumnInfo(name=self._config.ae.timestamp_column_name, dtype=datetime),
            ColumnInfo(name=self._config.ae.userid_column_name, dtype=str),
            ColumnInfo(name="accessdevicebrowser", dtype=str),
            ColumnInfo(name="accessdeviceos", dtype=str),
            ColumnInfo(name="authdevicename", dtype=str),
            ColumnInfo(name="result", dtype=bool),
            ColumnInfo(name="reason", dtype=str),
            # Derived columns
            IncrementColumn(name="logcount",
                            dtype=int,
                            input_name=self._config.ae.timestamp_column_name,
                            groupby_column=self._config.ae.userid_column_name),
            CustomColumn(name="locincrement",
                         dtype=int,
                         process_column_fn=partial(create_increment_col, column_name="location")),
        ]

        preprocess_schema = DataFrameInputSchema(column_info=preprocess_column_info, preserve_columns=["_batch_id"])

        schema = Schema(source=source_schema, preprocess=preprocess_schema)

        return schema
