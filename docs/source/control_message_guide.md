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

# Control Message Documentation

The control message is a JSON object used in the Morpheus pipeline workflow. It is wrapped in a `ControlMessage` object and passed between the Morpheus stages.

## Components

The control message has one main component: `inputs`. The inputs component is an array of input objects, each of which represents a separate input to the pipeline. Each input object has the following structure:

### Inputs
```
{
  "tasks": [
    // Array of task objects
  ],
  "metadata": {
    // Metadata object
  }
}
```

### Tasks

The tasks component of each input object is an array of task objects, each of which represents a separate task to be executed on the input data. Each task object has the following structure:

```
{
  "type": "string",
  "properties": {
    // Properties object
  }
}
```


- `type` : The type field of the task object is a string indicating the type of task to be executed. Currently, the following task types are supported:

  - `load` : Load input data from a specified file or files
  - `training` : Train a machine learning model on the input data
  - `inference` : Perform inference using a trained machine learning model on the input data

- `properties` : The properties field of the task object is an object containing task-specific properties. The specific properties required for each task type are described below.

  - The properties object for a `load` task has the following structure:
    ```
    {
      "loader_id": "string",
      "files": [
        "string"
      ]
    }
    ```

    - `loader_id` : The ID of the loader to be used for loading the input data. Currently, only the `fsspec` and `file_to_df` loader is supported. The user has the option to register custom loaders in the dataloader registry and utilize them in the pipeline.
    - `files` : An array of file paths or glob patterns specifying the input data to be loaded.

  - Incorporate key and value updates to properties objects as required for `training` and `inference` tasks. There is no specified format.

### Metadata
The metadata component of each input object is an object containing metadata information. Properties defined in this metadata component can be accessed anywhere across the stages that consume `ControlMessage` objects.

- `data_type` : which is a string indicating the type of data being processed. The supported data types are:
    - `payload` : Arbitrary input data
    - `Streaming` : Streaming data

## Example

This example demonstrates how to add various parameters to control message JSON. Below message contains an array of three task objects: a `load` task, a `training` task, and an `inference` task. The `load` task loads input data from two files specified in the files array to a dataframe using the fsspec loader. The `training` task trains a neural network model with three layers and ReLU activation. The `inference` task performs inference using the trained model with ID `model_001`. The metadata component of the input object indicates that the input data type is `payload`.

```json
{
    "inputs": [
        {
            "tasks": [
                {
                    "type": "load",
                    "properties": {
                        "loader_id": "fsspec",
                        "files": [
                            "/path/to/file1",
                            "/path/to/file2"
                        ]
                    }
                },
                {
                    "type": "training",
                    "properties": {
                        "model_type": "neural_network",
                        "model_params": {
                            "num_layers": 3,
                            "activation": "relu"
                        }
                    }
                },
                {
                    "type": "inference",
                    "properties": {
                        "model_id": "model_001"
                    }
                }
            ],
            "metadata": {
                "data_type": "payload"
            }
        }
    ]
}
```
