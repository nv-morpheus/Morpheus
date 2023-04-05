import io
import uuid
import cudf
import logging
import pandas as pd
import fsspec
import tempfile
import os
import shutil
from typing import Union, List, Tuple, Any

import torch
from torch.utils.data import Dataset


# TODO(Devin): Double check if we can use anything from deserializers or serializers code

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
                f"Invalid storage_type '{storage_type}' defaulting to 'in_memory', valid options are {self.VALID_STORAGE_TYPES}")

        if (file_format not in self.VALID_FILE_FORMATS):
            file_format = 'parquet'
            logging.warning(
                f"Invalid file_format '{file_format}' defaulting to 'parquet', valid options are {self.VALID_FILE_FORMATS}")

        self._file_format = file_format
        self._fs = fsspec.filesystem('file')
        self._sources = {}
        self._storage_dir = None
        self._storage_type = storage_type

        if (storage_type == 'filesystem'):
            self._storage_dir = tempfile.mkdtemp()

    def __contains__(self, item: Any) -> bool:
        return item in self._sources

    def __del__(self):
        if (self._storage_type == 'filesystem'):
            shutil.rmtree(self._storage_dir)

    def __len__(self) -> int:
        return len(self._sources)

    def _write_to_file(self, df: Union[cudf.DataFrame, pd.DataFrame], source_id: uuid.UUID) -> Any:
        if not isinstance(df, (cudf.DataFrame, pd.DataFrame)):
            raise ValueError("Invalid data source. Must be a cuDF or Pandas DataFrame.")

        if (self._storage_type == 'filesystem'):
            temp_path = os.path.join(self._storage_dir, f"{source_id}.{self._file_format}")
            if (self._file_format == 'parquet'):
                df.to_parquet(temp_path)
            elif (self._file_format == 'csv'):
                df.to_csv(temp_path)
        else:
            buf = io.BytesIO()
            if (self._file_format == 'parquet'):
                df.to_parquet(buf)
            elif (self._file_format == 'csv'):
                df.to_csv(buf)
            buf.seek(0)

            return buf

    @property
    def source(self) -> Union[List[io.BytesIO], List[str]]:
        if (self._storage_type == 'filesystem'):
            return [os.path.join(self._storage_dir, f"{source_id}.parquet") for source_id in self._sources]
        else:
            return list(self._sources.values())

    @property
    def storage_type(self) -> str:
        """
        Get the storage type used by the DataManager instance.

        :return: Storage type as a string.
        """

        return self._storage_type

    def get_num_rows(self, source_id: uuid.UUID) -> int:
        """
        Get the number of rows in a source given its source ID.
        :param source_id:
        :return:
        """
        _, num_rows = self._sources[source_id]
        return num_rows

    def load(self, id: uuid.UUID) -> cudf.DataFrame:
        """
        Load a cuDF DataFrame given a source ID.

        :param id: UUID of the source to be loaded.
        :return: Loaded cuDF DataFrame.
        """

        if (id not in self._sources):
            raise KeyError(f"Source ID '{id}' not found.")

        source, _ = self._sources[id]

        if (self._storage_type == 'in_memory'):
            buf = source
            buf.seek(0)
        elif (self._storage_type == 'filesystem'):
            buf = self._fs.open(source, 'rb')
        else:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

        if (self._file_format == 'parquet'):
            cudf_df = cudf.read_parquet(buf)
        elif (self._file_format == 'csv'):
            cudf_df = cudf.read_csv(buf)

        return cudf_df

    def store(self, source: Union[cudf.DataFrame, pd.DataFrame, str]) -> uuid.UUID:
        """
        Store a DataFrame or file path as a source and return the source ID.

        :param df: DataFrame or file path to store as a source.
        :return: UUID of the stored source.
        """

        source_id = uuid.uuid4()
        if (isinstance(source, (cudf.DataFrame, pd.DataFrame))):
            source_df = source
        else:
            source_df = cudf.read_parquet(source)

        num_rows = len(source_df)
        source = self._write_to_file(source_df, source_id)

        self._sources[source_id] = (source, num_rows)

        return source_id

    def remove(self, id: uuid.UUID) -> None:
        """
        Remove a source using its source ID.

        :param id: UUID of the source to be removed.
        """

        if (id in self._sources):
            del self._sources[id]


class DatasetFromDataManager(Dataset):
    """
    PyTorch Dataset class to load data from a DataManager instance.
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize the DatasetFromDataManager instance.
        :param data_manager:
        """

        self.data_manager = data_manager
        self.uuids = sorted(self.data_manager._sources.keys())
        self.num_rows_per_dataframe = [self.data_manager.get_num_rows(uuid) for uuid in self.uuids]
        self.total_rows = sum(self.num_rows_per_dataframe)

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find the corresponding dataframe based on the index
        for i, num_rows in enumerate(self.num_rows_per_dataframe):
            if index < num_rows:
                break
            index -= num_rows

        # Load the dataframe
        source_id = self.uuids[i]
        cudf_df = self.data_manager.load(source_id)

        # Assuming 'features' column contains input features and 'labels' column contains target labels
        features = torch.tensor(cudf_df['features'].to_array()[index], dtype=torch.float32)
        labels = torch.tensor(cudf_df['labels'].to_array()[index], dtype=torch.long)

        return features, labels
