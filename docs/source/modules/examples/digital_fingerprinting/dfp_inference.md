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

## DFP Inference Module

This module function performs the inference process.

### Configurable Parameters

- `model_name_formatter`: Formatter for model names (string).
- `fallback_username`: Fallback user to use if no model is found for a user (string).
- `timestamp_column_name`: Name of the timestamp column (string).

### Example JSON Configuration

```json
{
  "model_name_formatter": "user_{username}_model",
  "fallback_username": "generic_user",
  "timestamp_column_name": "timestamp"
}
```