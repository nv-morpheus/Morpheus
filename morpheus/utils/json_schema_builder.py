# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import json
from morpheus.utils.column_info import ColumnInfo
from morpheus.utils.column_info import DataFrameInputSchema
from morpheus.utils.column_info import DateTimeColumn
from morpheus.utils.column_info import DistinctIncrementColumn
from morpheus.utils.column_info import IncrementColumn
from morpheus.utils.column_info import RenameColumn
from morpheus.utils.column_info import StringCatColumn
from morpheus.utils.column_info import StringJoinColumn
from morpheus.utils.column_info import BoolColumn
from datetime import datetime
import warnings


class JSONSchemaBuilder:
    """Read and validate JSON defined schemas for Morpheus DataFrames. Construct DataFrameInputSchema from definition."""

    def __init__(self):
        """Initialize Class Attributes for Schema Processing"""

        self.supported_datatypes = {
            "datetime": datetime,
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
        }

        self.supported_column_types = {
            "ColumnInfo": self._build_ColumnInfo,
            "DateTimeColumn": self._build_DateTimeColumn,
            "RenameColumn": self._build_RenameColumn,
            "StringCatColumn": self._build_StringCatColumn,
            "StringJoinColumn": self._build_StringJoinColumn,
            "BoolColumn": self._build_BoolColumn,
            "IncrementColumn": self._build_IncrementColumn,
            "DistinctIncrementColumn": self._build_DistinctIncrementColumn,
        }

    def build_schema(self, json_file: str) -> DataFrameInputSchema:
        """
        Builds a DataFrameInputSchema objects from provided JSON schema defintion file.

        Parameters
        ----------
        json_file : str
            The path to the JSON file containing the schema definition

        Returns
        -------
        DataFrameInputSchema object
            Schema built and validated from the JSON file definnition
        """

        with open(json_file, "r") as f:
            self.schema = json.load(f)

        preserve_columns = self._load_preserve_cols()
        json_columns = self._load_json_cols()
        schema_list = self._build_colInfo_list()

        return DataFrameInputSchema(
            json_columns=json_columns,
            preserve_columns=preserve_columns,
            column_info=schema_list,
        )

    def _load_preserve_cols(self) -> list[str]:
        """Retrieve PRESERVE_COLUMNS list from schema. Returns empty list of not present."""

        return (
            self.schema["PRESERVE_COLUMNS"] if "PRESERVE_COLUMNS" in self.schema else []
        )

    def _load_json_cols(self) -> list[str]:
        """Retrieve JSON_COLUMNS list from schema. Returns empty list of not present."""

        return self.schema["JSON_COLUMNS"] if "JSON_COLUMNS" in self.schema else []

    def _build_colInfo_list(self) -> list[ColumnInfo]:
        """
        Builds list of ColumnInfo objects from the SCHEMA_COLUMNS attribute of the schema definition.

        Returns
        -------
        list of ColumnInfo object.

        Raises
        ------
        AttributeError of returned list will be empty.

        ValueError if unsupported column type is provided.
        """

        if "SCHEMA_COLUMNS" not in self.schema or self.schema["SCHEMA_COLUMNS"] == []:
            raise AttributeError(
                "SCHEMA_COLUMNS attribute must a be defined and non-empty list."
            )

        column_info_list = []

        for column in self.schema["SCHEMA_COLUMNS"]:

            column_type = column["type"]

            if column_type not in self.supported_column_types:
                raise ValueError(f"Unsupported column type: {column_type}.")

            # Call approrpiate builder function from class attributes
            column_info_list.append(self.supported_column_types[column_type](column))

        return column_info_list

    def _build_ColumnInfo(self, column: dict) -> ColumnInfo:
        """
        Builds ColumnInfo type column and returns the object.

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if "data_column" not in column or "dtype" not in column:
            raise AttributeError(
                "Both data_column and dtype attributes must be present in ColumnInfo definition."
            )

        data_column = column["data_column"]
        datatype = column["dtype"]

        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported dtype: {datatype}")

        return ColumnInfo(name=data_column, dtype=self.supported_datatypes[datatype])

    def _build_RenameColumn(self, column: dict) -> ColumnInfo:
        """
        Builds RenameColumn type column and returns the object.

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if "data_column" not in column or "dtype" not in column or "name" not in column:
            raise AttributeError(
                "data_column, name, and dtype attributes must be present in RenameColumn definition."
            )

        data_column = column["data_column"]
        datatype = column["dtype"]
        name = column["name"]

        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported dtype: {datatype}")

        return RenameColumn(name=name, dtype=datatype, input_name=data_column)

    def _build_DateTimeColumn(self, column: dict) -> ColumnInfo:
        """
        Builds DateTimeColumn type column and returns the object.

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if "data_column" not in column or "dtype" not in column or "name" not in column:
            raise AttributeError(
                "data_column, name, and dtype attributes must be present in DateTimeColumn definition."
            )

        data_column = column["data_column"]
        datatype = column["dtype"]
        name = column["name"]

        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported dtype: {datatype}")

        return DateTimeColumn(name=name, dtype=datatype, input_name=data_column)

    def _build_StringCatColumn(self, column: dict) -> ColumnInfo:
        """
        Builds StringCatColumn type column and returns the object.

        If "sep" is not specified in the schema file, default to ", "

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if (
            "data_columns" not in column
            or "dtype" not in column
            or "name" not in column
        ):
            raise AttributeError(
                "data_columns, name, and dtype attributes must be present in StringCatColumn definition."
            )

        sep = ", "

        if "sep" not in column:
            warnings.warn(
                "No separator provided for StringCatColumn. Defaulting to ', '."
            )
        else:
            sep = column["sep"]

        data_column = column["data_columns"]
        datatype = column["dtype"]
        name = column["name"]

        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported dtype: {datatype}")

        if not isinstance(data_column, list) or len(data_column) <= 1:
            raise ValueError("data_columns must be a list with more than one element.")

        return StringCatColumn(
            name=name, dtype=datatype, input_columns=data_column, sep=sep
        )

    def _build_StringJoinColumn(self, column: dict) -> ColumnInfo:
        """
        Builds StringJoinColumn type column and returns the object.

        If "sep" is not specified in the schema file, default to ", "

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if "data_column" not in column or "dtype" not in column or "name" not in column:
            raise AttributeError(
                "data_column, name, and dtype attributes must be present in StringJoinColumn definition."
            )

        sep = ", "

        if "sep" not in column:
            warnings.warn(
                "No separator provided for StringCatColumn. Defaulting to ', '."
            )
        else:
            sep = column["sep"]

        data_column = column["data_columns"]
        datatype = column["dtype"]
        name = column["name"]

        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported dtype: {datatype}")

        return StringJoinColumn(
            name=name, dtype=datatype, input_name=data_column, sep=sep
        )

    def _build_BoolColumn(self, column) -> ColumnInfo:
        """
        Builds BoolColumn type column and returns the object.

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if "data_column" not in column or "name" not in column:
            raise AttributeError(
                "data_column, and name attributes must be present in BoolColumn definition."
            )

        # Heirarchial check for value maps
        def is_dict_str_bool(d: Any) -> bool:
            if not isinstance(d, dict):
                return False
            return all(
                isinstance(key, str) and isinstance(value, bool)
                for key, value in d.items()
            )

        if "value_map" in column:
            if not is_dict_str_bool(column["value_map"]):
                raise ValueError("Value map must be of type Dict[str, bool].")

        elif "true_value" in column:
            if "false_value" in column:
                if not isinstance(column["true_value"], str) or not isinstance(
                    column["false_value"], str
                ):
                    raise ValueError(
                        "true_value and false_value must be string columns"
                    )

        elif "true_values" in column:
            if "false_values" in column:
                if not isinstance(column["true_value"], list) or not isinstance(
                    column["false_value"], list
                ):
                    raise ValueError("true_values and false_values must be lists")

        else:
            raise AttributeError(
                "One of value_map, (true_value, false_value), or (true_values, false_values) must be provided."
            )

        value_map = column["value_map"] if "value_map" in column else None
        true_value = column["true_value"] if "true_value" in column else None
        false_value = column["false_value"] if "false_value" in column else None
        true_values = column["true_values"] if "true_values" in column else None
        false_values = column["false_values"] if "false_values" in column else None
        data_column = column["data_columns"]
        datatype = bool
        name = column["name"]

        return BoolColumn(
            name=name,
            dtype=datatype,
            input_name=data_column,
            value_map=value_map,
            true_value=true_value,
            false_value=false_value,
            true_values=true_values,
            false_values=false_values,
        )

    def _build_IncrementColumn(self, column: dict) -> ColumnInfo:
        """
        Builds IncrementColumn type column and returns the object.

        If "sep" is not specified in the schema file, default to ", "

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if (
            "data_column" not in column
            or "dtype" not in column
            or "name" not in column
            or "groupby_column" not in column
        ):
            raise AttributeError(
                "data_column, name, groupby_column, and dtype attributes must be present in IncrementColumn definition."
            )

        period = "D"
        if "period" not in column:
            warnings.warn(
                "Group by period not specified for increment. Defaulting to 'D'"
            )
        else:
            period = column["period"]

        datatype = column["dtype"]
        name = column["name"]
        data_column = column["data_column"]
        groupby = column["groupby_column"]

        return IncrementColumn(
            name=name, dtype=datatype, input_name=data_column, groupby_column=groupby
        )

    def _build_DistinctIncrementColumn(self, column: dict) -> ColumnInfo:
        """
        Builds DistinctIncrementColumn type column and returns the object.

        If "sep" is not specified in the schema file, default to ", "

        Parameters
        ----------
        column: dict
            Dictionary of column schema from schema file

        Returns
        -------
        ColumnInfo object
            Constructed from Morpheus code for use in DataFrame Schema

        Raises
        ------
        AttributeError if require fields are not present

        ValueError if provided fields are not supported
        """

        if (
            "data_column" not in column
            or "dtype" not in column
            or "name" not in column
            or "groupby_column" not in column
            or "timestamp_column" not in column
        ):
            raise AttributeError(
                "data_column, name, groupby_column, timestamp_column, and dtype attributes must be present in DistinctIncrementColumn definition."
            )

        period = "D"
        if "period" not in column:
            warnings.warn(
                "Group by period not specified for increment. Defaulting to 'D'"
            )
        else:
            period = column["period"]

        datatype = column["dtype"]
        name = column["name"]
        data_column = column["data_column"]
        groupby = column["groupby_column"]
        ts = column["timestamp_column"]

        return DistinctIncrementColumn(
            name=name,
            dtype=datatype,
            input_name=data_column,
            groupby_column=groupby,
            timestamp_column=ts,
        )
