<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Morpheus CPU Only Mode
By default Morpheus is designed to take advantage of the GPU for accelerated processing. However, there are cases where you may want to run Morpheus on a system without access to a GPU. To address this need Morpheus provides a CPU only execution mode. Many stages within Morpheus require a GPU to run while others can opperate in both GPU and CPU execution mode. Attempting to add a GPU only stage to a pipeline that is configured to opperate in CPU only mode will result in an error.

## Execution Modes
By default Morpheus will run in GPU execution mode. Users have the choice of specifying the execution mode with either the Python API or from the command line.

### DataFrames and Tensors
With the selection of the execution mode comes an implies selection of DataFrame and tensor types. In GPU mode Morpheus will use cuDF DataFrames and CuPy tensors. In CPU mode Morpheus will use pandas DataFrames and NumPy tensors.

|Mode|DataFrame|Tensor|
| -- | ------- | ---- |
|GPU|cuDF|CuPy|
|CPU|pandas|NumPy|

### Python API
Execution modes are defined in the `morpheus.config.ExecutionMode` enum, which is then specified in the `execution_mode` attribute of the `morpheus.config.Config` object. The following example demonstrates how to explicitly set the execution mode of a pipeline to CPU only:

```python
from morpheus.config import Config
from morpheus.config import ExecutionMode
```

```python
config = Config()
config.execution_mode = ExecutionMode.CPU
```

> **Note**: The `execution_mode` and all other attributes of the `morpheus.config.Config` object must be set prior to constructing either the pipeline or any of the stage objects. The first time an instance of a `Config` object is used to construct an object will freeze the configuration and prevent any further changes.

#### Examples
The `examples/cpu_only`, `examples/developer_guide/1_simple_python_stage` and `examples/developer_guide/2_2_rabbitmq` examples demonstrate pipelines that are able to opperate in both GPU and CPU execution modes.

### Command Line
The `--use_cpu_only` flag is available as an option to the `morpheus run` subcommand.

```bash
morpheus run --use_cpu_only pipeline-other --help
```

#### Example
The following is a simple command line example of a pipeline that can execute in CPU only mode. To begin ensure that you have fetched the examples dataset by running the following command from the root of the Morpheus repository:
```bash
./scripts/fetch_data.py fetch examples
```

Then to run the pipeline run the following command:
```bash
morpheus --log_level=INFO \
  run --use_cpu_only pipeline-other \
  from-file --filename=examples/data/email_with_addresses.jsonlines \
  deserialize \
  monitor \
  serialize \
  to-file --filename=.tmp/output.jsonlines --overwrite
```
