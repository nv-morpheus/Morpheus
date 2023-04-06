import io
import cudf
import pandas as pd
import fsspec
from typing import Union, Any
import pyarrow.parquet as pq


class DataRecord:
    """Class for managing data records in different storage types and formats.

    Attributes:
        VALID_STORAGE_TYPES (tuple): Allowed storage types.
        VALID_FILE_FORMATS (tuple): Allowed file formats.
    """

    VALID_STORAGE_TYPES = ('in_memory', 'filesystem')
    VALID_FILE_FORMATS = ('parquet', 'csv')

    def __init__(self, data_source: Union[io.BytesIO, str],
                 data_label: str,
                 storage_type: str,
                 file_format: str,
                 copy_from_source: bool = False):
        """Initialize a DataRecord instance.

        Args:
            data_source (Union[io.BytesIO, str]): Data source, either a file path or a BytesIO object.
            data_label (str): Label for the data record.
            storage_type (str): Storage type, either 'in_memory' or 'filesystem'.
            file_format (str): File format, either 'parquet' or 'csv'.
            copy_from_source (bool, optional): If True, copy data from the source. Defaults to False.
        """

        self._copy_from_source = copy_from_source
        self._data = None
        self._data_label = data_label
        self._file_format = file_format
        self._fs = fsspec.filesystem('file')
        self._num_rows = 0
        self._owner = False
        self._backing_source = None
        self._storage_type = storage_type

        if self._file_format not in self.VALID_FILE_FORMATS:
            raise ValueError(f"Invalid file_format '{self._file_format}'")

        if self._storage_type not in self.VALID_STORAGE_TYPES:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

        if self._file_format == 'csv':
            self._data_reader = cudf.read_csv
        elif self._file_format == 'parquet':
            self._data_reader = cudf.read_parquet

        self._store(data_source)

    def __del__(self):
        """Delete the DataRecord instance."""
        try:
            if (self._owner and self._storage_type == 'filesystem'):
                self._fs.rm(self._backing_source)
        except Exception:
            pass

    def __len__(self):
        """Return the number of rows in the data record."""
        return self._num_rows

    def __repr__(self):
        """Return a string representation of the DataRecord instance."""
        return (f"DataRecord(data_label={self._data_label!r}, "
                f"storage_type={self._storage_type!r}, "
                f"file_format={self._file_format!r}, "
                f"num_rows={self._num_rows}, "
                f"owner={self._owner})")

    def __str__(self):
        """Return a string representation of the DataRecord instance."""
        return (f"DataRecord with label '{self._data_label}', "
                f"stored as {self._storage_type}, "
                f"file format: {self._file_format}, "
                f"number of rows: {self._num_rows}")

    @staticmethod
    def _read_source(file_path: Union[io.BytesIO, str], file_format: str = None) -> cudf.DataFrame:
        """Read data from a file path or BytesIO object.

        Args:
            file_path (Union[io.BytesIO, str]): File path or BytesIO object.
            file_format (str, optional): File format. Defaults to None.

        Returns:
            cudf.DataFrame: Data read from the source as a cuDF DataFrame.
        """

        _file_format = file_format or file_path.split('.')[-1]

        if _file_format == 'parquet':
            source_df = cudf.read_parquet(file_path)
        elif _file_format == 'csv':
            source_df = cudf.read_csv(file_path)
        else:
            raise ValueError(f"Unknown file format '{file_path}'")

        return source_df

    @staticmethod
    def _row_count_from_file(file_path, file_format=None) -> int:
        """Compute the number of rows in a Parquet or CSV file using pandas.

        Args:
            file_path (str): The path to the input file.
            file_format (str, optional): File format. Defaults to None.

        Returns:
            int: The number of rows in the file.
        """

        _file_format = file_format or file_path.split('.')[-1]

        if _file_format == 'parquet':
            # For Parquet files, use PyArrow to read the row count directly.
            par_file = pq.ParquetFile(file_path)
            row_count = par_file.metadata.num_rows
        elif _file_format == 'csv':
            # For CSV files, use pandas to read the file and count the rows.
            with pd.read_csv(file_path, chunksize=10 ** 6) as reader:
                row_count = sum(chunk.shape[0] for chunk in reader)

        else:
            raise ValueError(f"Unknown file format '{file_path}'")

        return row_count

    @property
    def backing_file(self) -> str:
        """Get the backing file for the data record.

        Returns:
            str: Backing file for the data record.
        """

        return self._backing_source

    @property
    def data(self):
        """Get the data associated with the data record.

        Returns:
            Any: Data associated with the data record.
        """

        return self._data

    @property
    def format(self):
        """
        Get the file format of the data record.
        Returns:
            str: File format of the data record.
        """
        return self._file_format

    @property
    def num_rows(self) -> int:
        """Get the number of rows in the data record.

        Returns:
            int: Number of rows in the data record.
        """

        return self._num_rows

    def _store(self, source: Union[io.BytesIO, str]) -> None:
        """Store the data in the appropriate storage type.

        Args:
            source (Union[io.BytesIO, str]): Data source, either a file path or a BytesIO object.
        """

        self._backing_source = "IO Buffer"

        if isinstance(source, (cudf.DataFrame, pd.DataFrame)):
            # source is a cuDF or Pandas DataFrame, so we need to write it to our temp storage
            self._num_rows = len(source)
            self._data = self._write_to_file(source, self._data_label)
            self._owner = True

        elif isinstance(source, str):
            if (self._storage_type == 'filesystem') and (not self._copy_from_source):
                # source is a file path, and we can use it directly
                self._data = source
                self._data_label = source
                self._num_rows = self._row_count_from_file(source)
                self._owner = False
            else:
                # source is a file path, and we need to copy it to our temp storage
                source_df = self._read_source(source)
                self._num_rows = len(source_df)
                self._data = self._write_to_file(source_df, self._data_label)
                self._owner = True
        else:
            raise ValueError("Invalid data source. Must be a cuDF or Pandas DataFrame or a file path.")

        if (self._storage_type == 'filesystem'):
            self._backing_source = self._data_label
        else:
            self._backing_source = "IO Buffer"

    def _write_to_file(self, df: Union[cudf.DataFrame, pd.DataFrame], data_label: str) -> Any:
        """Write the DataFrame to the specified storage type.

        Args:
            df (Union[cudf.DataFrame, pd.DataFrame]): DataFrame to write.
            data_label (str): Label for the data record.

        Returns:
            Any: Path or BytesIO object where the data is written.
        """

        if not isinstance(df, (cudf.DataFrame, pd.DataFrame)):
            raise ValueError("Invalid data source. Must be a cuDF or Pandas DataFrame.")

        if self._storage_type == 'filesystem':
            buf_or_path = data_label
        elif self._storage_type == 'in_memory':
            buf_or_path = io.BytesIO()
        else:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

        if self._file_format == 'parquet':
            df.to_parquet(buf_or_path)
        elif self._file_format == 'csv':
            df.to_csv(buf_or_path, index=False, header=True)

        return buf_or_path

    def load(self) -> cudf.DataFrame:
        """Load a cuDF DataFrame from the DataRecord.

        Returns:
            cudf.DataFrame: Loaded cuDF DataFrame.
        """

        if self._storage_type == 'in_memory':
            buf = self._data
            buf.seek(0)
        elif self._storage_type == 'filesystem':
            buf = self._fs.open(self._backing_source, 'rb')
        else:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

        cudf_df = self._data_reader(buf)

        return cudf_df
