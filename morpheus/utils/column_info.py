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

import pandas as pd

import cudf

logger = logging.getLogger(f"morpheus.{__name__}")

DEFAULT_DATE = '1970-01-01T00:00:00.000000+00:00'


# Note(Devin): Proxying this for backwards compatibility. Had to move the primary definition to avoid circular imports.
def process_dataframe(df_in: typing.Union[pd.DataFrame, cudf.DataFrame], input_schema) -> pd.DataFrame:
    """
    Processes a dataframe according to the given schema.

    Parameters
    ----------
    df_in : pandas.DataFrame or cudf.DataFrame
        The input dataframe to process.
    input_schema : object
        The schema used to process the dataframe.

    Returns
    -------
    pandas.DataFrame
        The processed dataframe.

    """

    from morpheus.utils import schema_transforms
    return schema_transforms.process_dataframe(df_in, input_schema)


def create_increment_col(df: pd.DataFrame,
                         column_name: str,
                         groupby_column: str = "username",
                         timestamp_column: str = "timestamp") -> pd.Series:
    """
    Create a new integer column counting unique occurrences of values in `column_name` grouped per-day using the
    timestamp values in `timestamp_column` and then grouping by `groupby_column` returning incrementing values starting
    at `1`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe.
    column_name : str
        Name of the column in which unique occurrences are counted.
    groupby_column : str, default "username"
        The column to group by.
    timestamp_column : str, default "timestamp"
        The column containing timestamp values.

    Returns
    -------
    pandas.Series
        The new column with incrementing values.

    """

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


def column_listjoin(df: pd.DataFrame, col_name: str) -> pd.Series:
    """
    Returns the array series `df[col_name]` as flattened string series.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe from which to get the column.
    col_name : str
        The column to transform.

    Returns
    -------
    pandas.Series
        A series with the arrays in the column flattened to strings.

    """

    if col_name in df:
        return df[col_name].transform(lambda x: ",".join(x)).astype('string')

    return pd.Series(None, dtype='string')




@dataclasses.dataclass
class ColumnInfo:
    """Defines a single column and type-cast."""
    name: str
    dtype: str  # The final type

    def get_pandas_dtype(self) -> str:
        """Return the pandas type string of column."""
        if ((isinstance(self.dtype, str) and self.dtype.startswith("datetime"))
                or (isinstance(self.dtype, type) and issubclass(self.dtype, datetime))):
            return "datetime64[ns]"

        if (isinstance(self.dtype, str)):
            return self.dtype

        return self.dtype.__name__

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

        return df[self.name].astype(self.get_pandas_dtype())


@dataclasses.dataclass
class CustomColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, defines a column to be computed by a user-defined function `process_column_fn`.

    Attributes
    ----------
    process_column_fn : Callable
        A function that takes a DataFrame and returns a Series.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Apply the `process_column_fn` to the DataFrame and return the result as a Series.

    """

    process_column_fn: typing.Callable

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the `process_column_fn` to the DataFrame and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to process.

        Returns
        -------
        pandas.Series
            The processed column.
        """
        return self.process_column_fn(df)


@dataclasses.dataclass
class RenameColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, adds the ability to also perform a rename.

    Attributes
    ----------
    input_name : str
        The name of the input column.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Rename the column and return it as a Series.

    """
    input_name: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Rename the column and return it as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The renamed column.
        """

        if (self.input_name not in df.columns):
            return pd.Series(None, index=df.index, dtype=self.get_pandas_dtype())

        return df[self.input_name].astype(self.get_pandas_dtype())


@dataclasses.dataclass
class BoolColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, adds the ability to map a set custom values as boolean values.

    Attributes
    ----------
    value_map : Dict[str, bool]
        A mapping from input values to boolean values.
    true_value : str
        The value to map to True.
    false_value : str
        The value to map to False.
    true_values : List[str]
        A list of values to map to True.
    false_values : List[str]
        A list of values to map to False.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Apply the mapping and return the result as a boolean Series.

    """

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
        """
        Apply the mapping and return the result as a boolean Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a boolean Series.
        """

        return super()._process_column(df).map(self.value_map).astype(bool)


@dataclasses.dataclass
class DateTimeColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, specific to casting UTC localized datetime values. When incoming values contain a
    time-zone offset string the values are converted to UTC, while values without a time-zone are assumed to be UTC.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Convert the values in the column to datetime and return the result as a Series.

    """

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert the values in the column to datetime and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a datetime Series.
        """

        return pd.to_datetime(super()._process_column(df), infer_datetime_format=True, utc=True)


@dataclasses.dataclass
class StringJoinColumn(RenameColumn):
    """
    Subclass of `RenameColumn`, converts incoming `list` values to string by joining by `sep`.

    Attributes
    ----------
    sep : str
        The separator to use when joining the values.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Join the values in the column and return the result as a Series.

    """

    sep: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Join the values in the column and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the column.

        Returns
        -------
        pandas.Series
            The processed column as a string Series.
        """

        return super()._process_column(df).str.join(sep=self.sep)


@dataclasses.dataclass
class StringCatColumn(ColumnInfo):
    """
    Subclass of `ColumnInfo`, concatenates values from multiple columns into a new string column separated by `sep`.

    Attributes
    ----------
    input_columns : List[str]
        The columns to concatenate.
    sep : str
        The separator to use when joining the values.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Concatenate the values from the input columns and return the result as a Series.

    """
    input_columns: typing.List[str]
    sep: str

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Concatenate the values from the input columns and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the columns.

        Returns
        -------
        pandas.Series
            The processed column as a string Series.
        """

        first_col = df[self.input_columns[0]]

        return first_col.str.cat(others=df[self.input_columns[1:]], sep=self.sep)


@dataclasses.dataclass
class IncrementColumn(DateTimeColumn):
    """
    Subclass of `DateTimeColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time
    window `period` based on dates in the `input_name` field.

    Attributes
    ----------
    groupby_column : str
        The column to group by.
    period : str
        The period to use when grouping.

    Methods
    -------
    _process_column(df: pandas.DataFrame) -> pandas.Series
        Count the unique occurrences and return the result as a Series.

    """

    groupby_column: str
    period: str = "D"

    def _process_column(self, df: pd.DataFrame) -> pd.Series:
        """
        Count the unique occurrences and return the result as a Series.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the columns.

        Returns
        -------
        pandas.Series
            The processed column as an integer Series.
        """

        period = super()._process_column(df).dt.to_period(self.period)

        # Create the `groupby_column`, per-period log count
        return df.groupby([self.groupby_column, period]).cumcount()


@dataclasses.dataclass
class DataFrameInputSchema:
    """
    Defines the schema specifying the columns to be included in the output `DataFrame`.

    Attributes
    ----------
    json_columns : List[str]
        The columns to include in the output DataFrame.
    column_info : List[ColumnInfo]
        Information about the columns.
    preserve_columns : List[str]
        The columns to preserve.
    row_filter : Callable[[pandas.DataFrame], pandas.DataFrame]
        A function to filter the rows of the DataFrame.

    Methods
    -------
    __post_init__()
        Compile the `preserve_columns` into a regular expression.

    """

    json_columns: typing.List[str] = dataclasses.field(default_factory=list)
    column_info: typing.List[ColumnInfo] = dataclasses.field(default_factory=list)
    preserve_columns: typing.List[str] = dataclasses.field(default_factory=list)
    row_filter: typing.Callable[[pd.DataFrame], pd.DataFrame] = None

    def __post_init__(self):
        """
        Compile the `preserve_columns` into a regular expression.
        """
        input_preserve_columns = self.preserve_columns

        # Ensure preserve_columns is a list
        if (not isinstance(input_preserve_columns, list)):
            input_preserve_columns = [input_preserve_columns]

        # Compile the regex
        if (input_preserve_columns is not None and len(input_preserve_columns) > 0):
            input_preserve_columns = re.compile(f"({'|'.join(input_preserve_columns)})")
        else:
            input_preserve_columns = None

        self.preserve_columns = input_preserve_columns
