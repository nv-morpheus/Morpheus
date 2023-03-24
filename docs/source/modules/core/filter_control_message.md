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

## Filter Control Message Module

When the requirements are met, this module gently discards the control messages.

### Configurable Parameters

- `enable_task_filtering` (bool): Enables filtering based on task type.
- `enable_data_type_filtering` (bool): Enables filtering based on data type.
- `filter_task_type` (str): The task type to be used as a filter.
- `filter_data_type` (str): The data type to be used as a filter.

### Example JSON Configuration

```json
{
  "enable_task_filtering": true,
  "enable_data_type_filtering": true,
  "filter_task_type": "specific_task",
  "filter_data_type": "desired_data_type"
}
```

### Default Settings

| Property                     | Value  |
| ----------------------------| -------|
| enable_data_type_filtering   | False  |
| enable_task_filtering        | False  |
