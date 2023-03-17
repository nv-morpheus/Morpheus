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

## Filter Detections Module

Filter message by a classification threshold.

The Filter Detections module is used to filter rows from a dataframe based on values in a tensor using a specified criteria. Rows in the `meta` dataframe are excluded if their associated value in the `probs` array is less than or equal to `threshold`.

### Configurable Parameters

- `field_name` (str): Name of the field to filter on. Defaults to `probs`.
- `threshold` (float): Threshold value to filter on. Defaults to `0.5`.
- `filter_source` (str): Source of the filter field. Defaults to `AUTO`.
- `copy` (bool): Whether to copy the rows or slice them. Defaults to `True`.
- `schema` (dict): Schema configuration.
    - `input_message_type` (str): Pickled message type.
    - `encoding` (str): Encoding used to pickle the message type.

### Example JSON Configuration

```json
{
  "field_name": "probs",
  "threshold": 0.5,
  "filter_source": "AUTO",
  "copy": true,
  "schema": {
    "input_message_type": "pickle_message_type",
    "encoding": "utf-8"
  }
}
```