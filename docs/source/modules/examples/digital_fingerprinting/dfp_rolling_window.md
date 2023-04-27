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

## DFP Split Users Module

This module function splits the data based on user IDs.

### Configurable Parameters

| Parameter             | Type   | Description                                                 | Example Value | Default Value |
|-----------------------|--------|-------------------------------------------------------------|---------------|---------------|
| cache_mode            | string | The user ID to use if the user ID is not found              | "batch"       | `batch`       |
| min_history           | int    | Minimum history to trigger a new training event             | 1             | `1`           |
| max_history           | int    | Maximum history to include in a new training event          | 0             | `0`           |
| timestamp_column_name | string | Name of the column containing timestamps                    | "timestamp"   | `timestamp`   |
| aggregation_span      | string | Lookback timespan for training data in a new training event | "60d"         | `60d`         |
| cache_to_disk         | bool   | Whether or not to cache streaming data to disk              | false         | `false`       |
| cache_dir             | string | Directory to use for caching streaming data                 | "./.cache"    | `./.cache`    |

### Example JSON Configuration

```json
{
  "cache_mode": "batch",
  "min_history": 1,
  "max_history": 0,
  "timestamp_column_name": "timestamp",
  "aggregation_span": "60d",
  "cache_to_disk": false,
  "cache_dir": "./.cache"
}
```
