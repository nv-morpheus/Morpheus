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

## Serialize Module

This module filters columns from a `MultiMessage` object, emitting a `MessageMeta`.

### Configurable Parameters

- `include` (str): Regex to include columns.
- `exclude` (List[str]): List of regex patterns to exclude columns.
- `fixed_columns` (bool): If true, the columns are fixed and not determined at runtime.
- `columns` (List[str]): List of columns to include.
- `use_cpp` (bool): If true, use C++ to serialize.

### Example JSON Configuration

```json
{
  "include": "^column",
  "exclude": ["column_to_exclude"],
  "fixed_columns": true,
  "columns": ["column1", "column2", "column3"],
  "use_cpp": true
}
```