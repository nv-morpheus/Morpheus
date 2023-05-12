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
import fsspec

from morpheus.io.data_storage_interface import RecordStorageInterface


class CSVRecordStorage(RecordStorageInterface):
    """Class for CSV record storage."""

    def save(self, record, data_label: str):
        """Save the provided record as a CSV file."""

        if self._storage_type == 'filesystem':
            self._backing_source = data_label
            record.to_csv(self._backing_source, index=False, header=True)
        elif self._storage_type == 'in_memory':
            buf = io.BytesIO()
            record.to_csv(buf, index=False, header=True)
            self._backing_source = buf
        else:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

    def load(self):
        """Load and return the saved CSV record."""

        if self._storage_type == 'in_memory':
            buf = self._backing_source
            buf.seek(0)
        elif self._storage_type == 'filesystem':
            buf = self._fs.open(self._backing_source, 'rb')
        else:
            raise ValueError(f"Invalid storage_type '{self._storage_type}'")

        return cudf.read_csv(buf)
