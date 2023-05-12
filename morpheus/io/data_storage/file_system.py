# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cudf
import fsspec

from typing import Union

import pandas as pd
import pyarrow.parquet as pq

from morpheus.io.data_storage_interface import RecordStorageInterface


def row_count_from_file(file_path, file_format_hint=None) -> int:
    """Compute the number of rows in a Parquet or CSV file using pandas.

    Args:
        file_path (str): The path to the input file.
        file_format_hint (str, optional): File format. Defaults to None.

    Returns:
        int: The number of rows in the file.
    """

    _file_format = file_format_hint or file_path.split('.')[-1]

    if (_file_format == 'parquet'):
        # For Parquet files, use PyArrow to read the row count directly.
        par_file = pq.ParquetFile(file_path)
        row_count = par_file.metadata.num_rows
    elif (_file_format == 'csv'):
        # For CSV files, use pandas to read the file and count the rows.
        with pd.read_csv(file_path, chunksize=10 ** 6) as reader:
            row_count = sum(chunk.shape[0] for chunk in reader)
    else:
        raise ValueError(f"Unknown file format '{file_path}'")

    return row_count


class FileSystemStorage(RecordStorageInterface):
    def __init__(self, data_label: str, file_format: str):
        super().__init__(data_label, file_format)

        self._fs = fsspec.filesystem('file')
        self._num_rows = 0

        if (self._file_format == 'csv'):
            self._data_reader = cudf.read_csv
            self._data_writer = lambda df, path: df.to_csv(path, index=False, header=True)
        elif (self._file_format == 'parquet'):
            self._data_reader = cudf.read_parquet
            self._data_writer = lambda df, path: df.to_parquet(path, index=False)
        else:
            raise NotImplementedError(f"File format {self._file_format} is not supported.")

    def _write_to_file(self, data_source: pd.DataFrame) -> str:
        # This method assumes data_source is either a BytesIO object or a file path
        file_path = f"{self._data_label}.{self._file_format}"

        self._data_writer(data_source, file_path)

        return file_path

    def delete(self) -> None:
        if (self._owner and self._backing_source is not None and self._fs.exists(self._backing_source)):
            self._fs.rm(self._backing_source)

    def load(self) -> cudf.DataFrame:
        return self._data_reader(self._backing_source)

    def store(self, data_source: Union[pd.DataFrame, cudf.DataFrame, str], copy_from_source: bool = False) -> None:
        # implementation goes here

        if (isinstance(data_source, cudf.DataFrame)):
            data_source = data_source.to_pandas()

        if (isinstance(data_source, pd.DataFrame)):
            self._backing_source = self._write_to_file(data_source)
            self._num_rows = len(data_source)
            self._owner = True
        elif (isinstance(data_source, str)):
            if (copy_from_source):
                data_source = self._data_reader(data_source)
                self._backing_source = self._write_to_file(data_source)
                self._num_rows = len(data_source)
                self._owner = True
            else:
                self._backing_source = data_source
                self._num_rows = row_count_from_file(data_source, self._file_format)
                self._owner = False

    @property
    def backing_source(self) -> str:
        return self._backing_source

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def owner(self) -> bool:
        return self._owner
