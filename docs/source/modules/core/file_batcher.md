<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## File Batcher Module

This module loads the input files, removes files that are older than the chosen window of time, and then groups the remaining files by period that fall inside the window.

### Configurable Parameters

- `batching_options`: A dictionary containing the following options:
    - `end_time`: End time of the time window (datetime or string format)
    - `iso_date_regex_pattern`: Regex pattern for ISO date matching
    - `parser_kwargs`: Additional arguments for the parser (dictionary)
    - `period`: Time period for grouping files (e.g., '1d' for 1 day)
    - `sampling_rate_s`: Sampling rate in seconds (integer)
    - `start_time`: Start time of the time window (datetime or string format)
- `cache_dir`: Cache directory (string)
- `file_type`: File type (string)
- `filter_nulls`: Whether to filter null values (boolean)
    - `True`: Filter null values
    - `False`: Don't filter null values
- `schema`: Data schema (dictionary)
- `timestamp_column_name`: Name of the timestamp column (string)

### Example JSON Configuration

```json
{
  "batching_options": {
    "end_time": "2023-03-14T23:59:59",
    "iso_date_regex_pattern": "\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    "parser_kwargs": {},
    "period": "1d",
    "sampling_rate_s": 60,
    "start_time": "2023-03-01T00:00:00"
  },
  "cache_dir": "./file_batcher_cache",
  "file_type": "csv",
  "filter_nulls": true,
  "schema": {
    "column1": "string",
    "column2": "int",
    "timestamp": "datetime"
  },
  "timestamp_column_name": "timestamp"
}
```