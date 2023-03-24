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

## File to DataFrame Loader

This function is used to load files containing data into a dataframe. Dataframe is created by processing files either using a single thread, multiprocess, dask, or dask_thread. This the function determines the download method to use, and if it starts with "dask," it creates a dask client and uses it to process the files. Otherwise, it uses a single thread or multiprocess to process the files. This function then caches the resulting dataframe using a hash of the file paths. In addition to loading data from the disk, it has the ability to load the file content from S3 buckets.

**Note** :  Loaders receive configuration from `load` task via the [control message](./../../source/control_message_guide.md) during runtime.

### Configurable Parameters

- `id` (str): Registered loader id.

### Example JSON Configuration

```json
{
	"loaders": [{
		"id": "file_to_df"
	}]
}
```

### Default Settings

| Property                | Value      |
| -----------------------| ----------|
| cache_dir               | ./.cache  |
| file_type               | JSON      |
| filter_null             | False     |
| parser_kwargs           | None      |
| timestamp_column_name   | timestamp |

**Note** : The [file_batcher](../../../../morpheus/modules/file_batcher.py) module currently generates tasks internally and assigns them to control messages, and then sends them to a [file_to_df_loader](../../../../morpheus/loaders/file_to_df_loader.py). Having stated that, this loader's configuration is obtained from the [File Batcher](../../modules/core/file_batcher.md) module configuration.
