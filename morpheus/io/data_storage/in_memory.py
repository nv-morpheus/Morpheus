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
import io
import pandas as pd

from typing import Union

from morpheus.io.data_storage_interface import RecordStorageInterface


class InMemoryStorage(RecordStorageInterface):
    def __init__(self, data_label: str, file_format: str):
        super().__init__(data_label, file_format)

        self._backing_source = "IO Buffer"
        self._data = io.BytesIO()
        self._num_rows = 0
        self._owner = True

        if (self._file_format == 'csv'):
            self._data_reader = pd.read_csv
            self._data_writer = lambda df, buffer: df.to_csv(buffer, index=False, header=True)
        elif (self._file_format == 'parquet'):
            self._data_reader = pd.read_parquet
            self._data_writer = lambda df, buffer: df.to_parquet(buffer, index=False)
        else:
            raise NotImplementedError(f"File format {self._file_format} is not supported.")

    def load(self) -> pd.DataFrame:
        self._data.seek(0)
        return self._data_reader(self._data)

    def delete(self) -> None:
        if self._owner:
            self._data.close()

    def store(self, data_source: Union[pd.DataFrame, cudf.DataFrame, str], copy_from_source: bool = False) -> None:
        # implementation goes here

        self._data = io.BytesIO()
        if (isinstance(data_source, str)):
            data_source = self._data_reader(data_source)

        self._data_writer(data_source, self._data)
        self._num_rows = len(data_source)

    @property
    def backing_source(self) -> str:
        return self._backing_source

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def owner(self) -> bool:
        return self._owner
