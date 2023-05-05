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

## Multiplexer Module

The multiplexer receives data packets from one or more input ports and interleaves them into a single output.

### Configurable Parameters

| Parameter                 | Type      | Description                                                                                               | Example Value | Default Value |
|---------------------------|-----------|-----------------------------------------------------------------------------------------------------------|---------------|---------------|
| `num_input_ports_to_merge`| integer   | Number of nodes stream data to be combined                                                      | `3`        | `2`        |
| `stop_after_secs`         | integer   | Time in seconds to halt the process                     | `100`          | `-1`          |
| `streaming`         | boolean   | Execution in streaming mode is indicated by this flag                     | `True`          | `False`          |

### Example JSON Configuration

```json
{
  "num_input_ports_to_merge": 3,
  "stop_after_secs": -1,
  "streaming": true
}
