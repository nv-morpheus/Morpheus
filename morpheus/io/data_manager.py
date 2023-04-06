import uuid
import cudf
import logging
import pandas as pd
import fsspec
import tempfile
import os
import shutil
from typing import Union, Optional, List, Tuple, Any

import torch
from torch.utils.data import Dataset

from morpheus.io.data_record import DataRecord


class DataManager():
    """
    DataManager class to manage the storage and retrieval of files
    using either in-memory or filesystem storage.
    """

    VALID_STORAGE_TYPES = ('in_memory', 'filesystem')
    VALID_FILE_FORMATS = ('parquet', 'csv')

    def __init__(self, storage_type: str = 'in_memory', file_format: str = 'parquet'):
        """
        Initialize the DataManager instance.

        :param storage_type: Specifies the storage type to be used. Can be either 'in_memory' or 'filesystem'.
        :param file_format: Specifies the file format to be used. Can be either 'parquet' or 'csv'.
        """

        if (storage_type not in self.VALID_STORAGE_TYPES):
            storage_type = 'in_memory'
            logging.warning(
                f"Invalid storage_type '{storage_type}' defaulting to 'in_memory', valid options are "
                f"{self.VALID_STORAGE_TYPES}")

        if (file_format not in self.VALID_FILE_FORMATS):
            file_format = 'parquet'
            logging.warning(
                f"Invalid file_format '{file_format}' defaulting to 'parquet', valid options are "
                f"{self.VALID_FILE_FORMATS}")

        self._file_format = file_format
        self._fs = fsspec.filesystem('file')
        self._manifest = {}
        self._records = {}
        self._storage_dir = None
        self._storage_type = storage_type
        self._total_rows = 0

        if (storage_type == 'filesystem'):
            self._storage_dir = tempfile.mkdtemp()

        if (file_format == 'parquet'):
            self._data_reader = cudf.read_parquet
        elif (file_format == 'csv'):
            self._data_reader = cudf.read_csv
        else:
            raise ValueError(f"Invalid file_format '{self._file_format}'")

    def __contains__(self, item: Any) -> bool:
        return item in self._records

    def __del__(self):
        if (self._storage_type == 'filesystem'):
            shutil.rmtree(self._storage_dir)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self):
        return (f"DataManager(records={self.num_rows}, "
                f"storage_type={self._storage_type!r}, "
                f"storage directory={self._storage_dir!r})")

    def __str__(self):
        return (f"DataManager with {self.num_rows} records, "
                f"storage type: {self._storage_type}, "
                f"storage directory: {self._storage_dir}")

    def _update_manifest(self, source_id: uuid.UUID, action: str) -> None:
        data_record = self._records[source_id]

        if action == 'store':
            self._manifest[source_id] = data_record.data
            self._total_rows += data_record.num_rows
        elif action == 'remove':
            self._total_rows -= data_record.num_rows
            del self._manifest[source_id]

    @property
    def manifest(self) -> dict:
        """
        Retrieve a mapping of UUIDs to their filenames or labels.

        :return: A dictionary containing UUID to filename/label mappings.
        """

        return self._manifest

    @property
    def num_rows(self) -> int:
        """
        Get the number of rows in a source given its source ID.
        :param source_id:
        :return:
        """

        return self._total_rows

    @property
    def records(self):
        return self._records

    @property
    def storage_type(self) -> str:
        """
        Get the storage type used by the DataManager instance.

        :return: Storage type as a string.
        """

        return self._storage_type

    def get_record(self, source_id: uuid.UUID) -> DataRecord:
        """
        Get a DataRecord instance given a source ID.

        :param source_id: UUID of the source to be retrieved.
        :return: DataRecord instance.
        """

        if source_id not in self._records:
            raise KeyError(f"Source ID '{source_id}' not found.")

        return self._records[source_id]

    def load(self, source_id: uuid.UUID) -> cudf.DataFrame:
        """
        Load a cuDF DataFrame given a source ID.

        :param source_id: UUID of the source to be loaded.
        :return: Loaded cuDF DataFrame.
        """

        if source_id not in self._records:
            raise KeyError(f"Source ID '{source_id}' not found.")

        data_record = self._records[source_id]

        return data_record.load()

    def store(self, data_source: Union[cudf.DataFrame, pd.DataFrame, str], copy_from_source: bool = False,
              data_label: Optional[str] = None) -> uuid.UUID:
        """
        Store a DataFrame or file path as a source and return the source ID.

        :param data_source: DataFrame or file path to store as a source.
        :param copy_from_source: Whether to copy the data on disk when the input is a file path and the storage type is
        'filesystem'.
        :param data_label: Optional label for the stored data.
        :return: UUID of the stored source.
        """

        tracking_id = uuid.uuid4()

        if (self._storage_type == 'filesystem'):
            data_label = os.path.join(self._storage_dir, f"{tracking_id}.{self._file_format}")
        else:
            data_label = data_label or f'dataframe_{tracking_id}'

        data_record = DataRecord(data_source=data_source, data_label=data_label, storage_type=self.storage_type,
                                 file_format=self._file_format, copy_from_source=copy_from_source)

        self._records[tracking_id] = data_record
        self._update_manifest(tracking_id, action='store')

        return tracking_id

    def remove(self, source_id: uuid.UUID) -> None:
        """
        Remove a source using its source ID.

        :param source_id: UUID of the source to be removed.
        """

        if (source_id not in self._records):
            raise KeyError(f"Source ID '{source_id}' does not exist.")

        self._update_manifest(source_id, 'remove')
        del self._records[source_id]

