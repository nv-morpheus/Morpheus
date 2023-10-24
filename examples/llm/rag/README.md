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

# Retrieval Augmented Generation (RAG) Pipeline

## Table of Contents

## Background Information

### Purpose

The purpose of this example is to illustrate how a user could build a pipeline which will integrate an LLM service
into a Morpheus pipeline. This example builds on the previous example [completion pipeline](../completion/README.md),
by adding the ability to augment LLM queries with context information from a knowledge base. Appending this context
helps improve the responses from the LLM by providing additional background contextual and factual information which
the LLM can pull from for its response.

### Source Documents

- In order for this pipeline to function correctly, a Vector Database must already have been populated with information
  that can be retrieved.
- An example of populating a database is illustrated in [FEA]: Create Sherlock example for VDB Upload #1298.
- This example assumes that pipeline has already been run to completion.

### Embedding Model

- This pipeline can support any type of embedding model that can convert text into a vector of floats.
- For the example, we will use all-MiniLM-L6-v2. It is small, accurate, and included in the Morpheus repo via LFS;
  it is also the default model used in the [VDB upload](../vdb_upload/README.md) pipeline.

### Vector Database Service

- Any vector database can be used to store the resulting embedding and corresponding metadata.
- It would be trivial to update the example to use Chroma or FAISS if needed.
- For this example, we will be using Milvus since it is the default VDB used in
  the [VDB upload](../vdb_upload/README.md) pipeline.

### Implementation and Design Decisions

### Implementation Details

[Original GitHub issue](https://github.com/nv-morpheus/Morpheus/issues/1306)

**TODO**

### Rationale Behind Design Decisions

**TODO**

### Standalone Morpheus Pipeline

**TODO**

The standalone Morpheus pipeline is built using the following components:

- An InMemorySourceStage to hold the LLM queries in a DataFrame.
- A DeserializationStage to convert the MessageMeta objects into ControlMessages needed by the LLMEngine.
    - New functionality was added to the DeserializeStage to support ControlMessages and add a default task to each
      message.
- A LLMEngineStage then wraps the core LLMEngine functionality.
    - An ExtracterNode pulls the questions out of the DataFrame.
    - A RAGNode performs the retrieval and adds the context to the query using the supplied template and executes the
      LLM.
    - Finally, the responses are put back into the ControlMessage using a SimpleTaskHandler.
- The pipeline concludes with an InMemorySink stage to store the results.

> **Note:** For this to function correctly, the VDB upload pipeline must have been run previously.

### Persistent Morpheus Pipeline

The persistent Morpheus pipeline is functionally similar to the standalone pipeline, but it uses multiple sources and
multiple sinks to perform both the upload and retrieval portions in the same pipeline. The benefit of this pipeline over
the standalone pipeline is that no VDB upload process needs to be run beforehand. Everything runs in a single pipeline.

### Getting Started

### Prerequisites

Before running the pipeline, we need to ensure that the following services are running:

#### Milvus Service

- Follow the instructions [here](https://milvus.io/docs/install_standalone-docker.md) to install and run a Milvus
  service.

#### Triton Service

- Pull the Docker image for Triton:
  ```bash
  docker pull nvcr.io/nvidia/tritonserver:23.06-py3
  ```

- From the Morpheus repo root directory, run the following to launch Triton and load the `all-MiniLM-L6-v2` model:
  ```bash
  docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002
   -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver 
   --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit
   --load-model all-MiniLM-L6-v2
  ```

  This will launch Triton and only load the `all-MiniLM-L6-v2` model. Once Triton has loaded the model, the following
  will be displayed:
    ```
    +------------------+---------+--------+
    | Model            | Version | Status |
    +------------------+---------+--------+
    | all-MiniLM-L6-v2 | 1       | READY  |
    +------------------+---------+--------+
    ```

### Running the Morpheus Pipeline
The top level entrypoint to each of the LLM example pipelines is `examples/llm/main.py`. This script accepts a set
of Options and a Pipeline to run. Baseline options are below, and for the purposes of this document we'll assume a
pipeline option of `rag`:

### Run example:

```bash
python examples/llm/main.py [OPTIONS...] rag [ACTION] --model_name all-MiniLM-L6-v2
```

### Options:

- `--log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]`
    - **Description**: Specifies the logging level.
    - **Default**: `INFO`

- `--use_cpp BOOLEAN`
    - **Description**: Opt to use C++ node and message types over python. Recommended only in case of bugs.
    - **Default**: `False`

- `--version`
    - **Description**: Display the script's current version.

- `--help`
    - **Description**: Show the help message with options and commands details.

### Commands:

- ... other pipelines ...
- `rag`

---

## Options for `rag` Command

The `rag` command has its own set of options and commands:

### Commands:

- `persistant`
- `pipeline`

