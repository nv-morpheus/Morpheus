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

## WriteToFile Module

This module writes messages to a file.

### Configurable Parameters

- `filename` (str): Path to the output file.
- `overwrite` (bool): If true, overwrite the file if it exists.
- `flush` (bool): If true, flush the file after each write.
- `file_type` (FileTypes): Type of file to write.
- `include_index_col` (bool): If true, include the index column.

### Example JSON Configuration

```json
{
  "filename": "output.csv",
  "overwrite": true,
  "flush": false,
  "file_type": "CSV",
  "include_index_col": false
}
```

### Default Settings

| Property             | Value          |
| --------------------| -------------- |
| file_type            | FileTypes.Auto |
| filename             | None           |
| flush                | False          |
| include_index_col    | True           |
| overwrite            | False          |
