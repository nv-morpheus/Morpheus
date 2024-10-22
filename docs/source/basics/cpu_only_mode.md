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

# Morpheus CPU-Only Mode
By default Morpheus is designed to take advantage of the GPU for accelerated processing. However, there are cases where you may want to run Morpheus on a system without access to a GPU. To address this need Morpheus provides a CPU only execution mode. Many stages within Morpheus require a GPU to run while others can operate in both GPU and CPU execution mode. Attempting to add a GPU only stage to a pipeline that is configured to operate in CPU only mode will result in an error.

## Execution Modes
By default Morpheus will run in GPU execution mode. Users have the choice of specifying the execution mode with either the Python API or from the command line.

## CPU Only Pipelines
### Python API
Execution modes are defined in the `morpheus.config.ExecutionMode` enumeration, which is then specified in the `execution_mode` attribute of the `morpheus.config.Config` object. The following example demonstrates how to explicitly set the execution mode of a pipeline to CPU only:

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
The `examples/cpu_only`, `examples/developer_guide/1_simple_python_stage` and `examples/developer_guide/2_2_rabbitmq` examples demonstrate pipelines that are able to operate in both GPU and CPU execution modes.

### Command Line
The `--use_cpu_only` flag is available as an option to the `morpheus run` sub-command.

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

## Designing Stages for CPU Execution
It is up to the author of each stage to decide which exectution modes are supported. Options are: CPU, GPU or both. As mentioned previously the default execution mode is GPU, authors of stages which require a GPU do not need to make any changes to their stage definitions.

### DataFrames and Tensors
With the selection of the execution mode comes an implies selection of DataFrame and tensor types. In GPU mode Morpheus will use cuDF DataFrames and CuPy tensors. In CPU mode Morpheus will use pandas DataFrames and NumPy tensors.

|Mode|DataFrame|Tensor|
| -- | ------- | ---- |
|GPU|[cuDF](https://docs.rapids.ai/api/cudf/stable/)|[CuPy](https://cupy.dev/)|
|CPU|[pandas](https://pandas.pydata.org/)|[NumPy](https://numpy.org/)|

### Stages defined with `@stage` and `@source` decorators
Both the `@stage` and `@source` decorators have an optional `execution_modes` parameter that accepts a tuple of `morpheus.config.ExecutionMode` values which is used to specify the supported executions mode of the stage.

#### CPU-only Source & Stage Examples
```python
import logging

import pandas as pd

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(f"morpheus.{__name__}")

@source(execution_modes=(ExecutionMode.CPU, ))
def simple_source(num_rows: int = 10) -> MessageMeta:
    df = pd.DataFrame({"a": range(num_rows)})
    message = MessageMeta(df)
    yield message

@stage(execution_modes=(ExecutionMode.CPU, ))
def print_msg(msg: MessageMeta) -> MessageMeta:
    logger.info(f"Receive a message with a DataFrame of type: {type(msg.df)}")
    return msg

def main():
    configure_logging(log_level=logging.INFO)
    pipeline = LinearPipeline(config)
    pipeline.set_source(simple_source(config))
    pipeline.add_stage(print_msg(config))
    pipeline.run()

if __name__ == "__main__":
    main()
```

#### CPU & GPU Source & Stage Examples
Supporting both CPU and GPU execution modes requires writing code that can handle both types of DataFrames and tensors. In many cases code designed to work with pandas will work with cuDF, and code designed to work with Numpy will work with CuPy without requiring any changes to the code. In some cases however, the API may differ slightly and there is a need to know the payload type, care must be taken not to directly import `cudf` or any other package requiring a GPU when running in CPU mode on a system without a GPU. Morpheus provides some helper methods to assist with this in the {py:mod}`~morpheus.utils.type_utils` module, such as {py:func}`~morpheus.utils.type_utils.is_cudf_type` and {py:func}`~morpheus.utils.type_utils.get_df_pkg_from_obj`.
