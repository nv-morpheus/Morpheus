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

## MLflow Model Writer Module

This module uploads trained models to the MLflow server.

### Configurable Parameters

- `model_name_formatter` (str): Formatter for the model name.
- `experiment_name_formatter` (str): Formatter for the experiment name.
- `conda_env` (str): Conda environment for the model.
- `timestamp_column_name` (str): Name of the timestamp column.
- `databricks_permissions` (dict): Permissions for the model.

### Example JSON Configuration

```json
{
  "model_name_formatter": "model_name_{timestamp}",
  "experiment_name_formatter": "experiment_name_{timestamp}",
  "conda_env": "path/to/conda_env.yml",
  "timestamp_column_name": "timestamp",
  "databricks_permissions": {
    "read": ["read_user1", "read_user2"],
    "write": ["write_user1", "write_user2"]
  }
}
```