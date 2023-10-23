## Table of Contents

1. [Background Information](#Background-Information)
    - [Purpose](#Purpose)
    - [Source Documents](#Source-Documents)
    - [Embedding Model](#Embedding-Model)
    - [Vector Database Service](#Vector-Database-Service)
2. [Implementation and Design Decisions](#Implementation-and-Design-Decisions)
3. [Getting Started](#Getting-Started)
    - [Prerequisites](#Prerequisites)
        - [Milvus Service](#Milvus-Service)
        - [Triton Service](#Triton-Service)
    - [Running the Morpheus Pipeline](#Running-the-Morpheus-Pipeline)
    - [Options for vdb_upload Command](#Options-for-vdb_upload-Command)
    - [Exporting and Deploying a Different Model from Huggingface](#Exporting-and-Deploying-a-Different-Model-from-Huggingface)


## Background Information

### Purpose

The primary objective of this example is to demonstrate the construction of a pipeline that performs the following
tasks:

- Accepts a collection of documents.
- Divides the documents into smaller segments or chunks.
- Computes the embedding vector for each of these chunks.
- Uploads the text chunks along with their associated embeddings to a Vector Database (VDB).

### Source Documents

- The pipeline is designed to process text-based input from various document types. Possible use cases could
  include structured documents like PDFs, dynamic sources such as web pages, and image-based documents through future
  Optical Character Recognition (OCR) integration.

- For this demonstration, the source documents are obtained from RSS feeds combined with a web scraper. The rationale
  behind this selection includes:
    - Emulating practical cyber scenarios: Cybersecurity RSS feeds can serve as the foundation for a comprehensive
      knowledge database, such as for a security chatbot.
    - Minimizing external dependencies: Relying on RSS feeds and web scraping avoids the need for specialized datasets
      or API keys.

### Embedding Model

- The pipeline can accommodate various embedding models that transform text into vectors of floating-point numbers.
  Several models from Huggingface, such as `paraphrase-multilingual-mpnet-base-v2`, `e5-large-v2`,
  and `all-mpnet-base-v2`, have been evaluated for compatibility.

- For the purposes of this demonstration, the model `all-MiniLM-L6-v2` is employed. This model is included via LFS
  in this repository, and was chosen for its efficiency and compactness, characterized by a smaller embedding dimension
  of 384.

### Vector Database Service

- The architecture is agnostic to the choice of Vector Database (VDB) for storing embeddings and their metadata. While
  the present implementation employs Milvus due to its GPU-accelerated indices, the design supports easy integration
  with other databases like Chroma or FAISS, should the need arise.

## Implementation and Design Decisions

- A detailed explanation of the implementation.
- Rationale behind design decisions, possibly citing the GitHub issue for context.

## Getting Started

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
pipeline option of `vdb_upload`:

### Run example:

```bash
python examples/llm/main.py [OPTIONS...] vdb_upload [ACTION] --model_name all-MiniLM-L6-v2
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
- `vdb_upload`

---

## Options for `vdb_upload` Command

The `vdb_upload` command has its own set of options and commands:

### Commands:

- `export-triton-model`
- `langchain`
- `pipeline`

### Exporting and Deploying a Different Model from Huggingface

If you're looking to incorporate a different embedding model from Huggingface into the pipeline, follow the steps below
using `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` as an example:

1. **Identify the Desired Model**:
    - Head over to the [Huggingface Model Hub](https://huggingface.co/models) and search for the model you want. For
      this example, we are looking at `e5-large-v2`.

2. **Run the Pipeline Call with the Chosen Model**:
    - Execute the following command with the model name you've identified:
      ```bash
      python examples/llm/main.py vdb_upload export-triton-model  --model_name 
       sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --triton_repo ./models/triton-model-repo
      ```

3. **Handling Unauthorized Errors**:
    - Please ensure you provide the correct model name. A common pitfall is encountering an `unauthorized error`. If
      you see the following error:
      ```bash
      requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url:
      ```
      This typically means the model name you provided does not match the one available on Huggingface. Double-check
      the model name and try again.

4. **Confirm Successful Model Export**:
    - After running the command, ensure that the specified `--triton_repo` directory now contains the exported model in
      the correct format, ready for deployment.
    ```bash
    $ ls ${MORPHEUS_ROOT}/models/triton-model-repo | grep paraphrase-multilingual-mpnet-base-v2
    
    paraphrase-multilingual-mpnet-base-v2 
    ```

5. **Deploy the Model**:
    - Reload the docker container, specifying that we also need to load paraphrase-multilingual-mpnet-base-v2
    ```bash
    docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 
     -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver 
     --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model 
     all-MiniLM-L6-v2 --load-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    ```

    - You should see seomthing similar to the following, indicating Triton has succesfully loaded the model:
    ```shell
    +----------------------------------+------------------------------------------------------------------------------------------+
    | Option                           | Value                                                                                    |
    +----------------------------------+------------------------------------------------------------------------------------------+
    | server_id                        | triton                                                                                   |
    | server_version                   | 2.35.0                                                                                   |
    | server_extensions                | classification sequence model_repository ... schedule_policy                             |
    | model_repository_path[0]         | /models/triton-model-repo                                                                |
    | model_control_mode               | MODE_EXPLICIT                                                                            |
    | startup_models_0                 | all-MiniLM-L6-v2                                                                         |
    | startup_models_1                 | paraphrase-multilingual-mpnet-base-v2                                                    |
    | strict_model_config              | 0                                                                                        |
    | rate_limit                       | OFF                                                                                      |
    | pinned_memory_pool_byte_size     | 268435456                                                                                |
    | cuda_memory_pool_byte_size{0}    | 67108864                                                                                 |
    | cuda_memory_pool_byte_size{1}    | 67108864                                                                                 |
    | min_supported_compute_capability | 6.0                                                                                      |
    | strict_readiness                 | 1                                                                                        |
    | exit_timeout                     | 30                                                                                       |
    | cache_enabled                    | 0                                                                                        |
    +----------------------------------+------------------------------------------------------------------------------------------+
    ```
6. **Update the Pipeline Call**:
    - Now that the model has been exported and deployed, we can update the pipeline call to use the new model:
    ```bash
    python examples/llm/main.py vdb_upload pipeline --model_name paraphrase-multilingual-mpnet-base-v2
    ```

### Running the Langchain Pipeline (Optional)

- Optional guide for running the Langchain pipeline, if applicable.## Developer Docs

- A link to the developer documentation where the README.md is also linked.

> **Note**: This pipeline will, by default, run continuously repeatedly polling the configured RSS sources. To run for a
> fixed number of iterations, add the `--stop_after=N` flag.
