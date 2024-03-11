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

# Cyber Developer Day 2024

## Prerequisites

To run the Cyber Developer Day 2024, you will need to have the access to the NeMo Inference Service and NVIDIA AI Foundation Models API. These are necessary to support running LLMs which are the focus of the Cyber Developer Day.

### NVIDIA GPU Cloud

To access the NeMo Inference Service, you will need to have the following environment variables set: `NGC_API_KEY`. To obtain the `NGC_API_KEY`, please visit the [NGC website](https://ngc.nvidia.com/) for [instructions](https://docs.nvidia.com/ngc/gpu-cloud/ngc-user-guide/index.html#generating-api-key) on generating your API key. **Note:** You must belong to an organization which has access to NeMo LLM in order for your NGC API key to work.

### NVIDIA AI Foundation Models

To access the NVIDIA AI Foundation Models API, you will need to have the following environment variables set: `NVIDIA_API_KEY`. To obtain your `NVIDIA_API_KEY`, use the following steps:
1. Create a free account with the [NVIDIA GPU Cloud](https://catalog.ngc.nvidia.com/) service, which hosts AI solution catalogs, containers, models, etc.
2. Navigate to `Catalog > AI Foundation Models > (Model with API endpoint)`.
3. Select the `API` option and click `Generate Key`.
4. Save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

### Creating an Environment File

To automatically use these API keys, you can create an environment file located at ${MORPHEUS_ROOT}/examples/cyber_dev_day with the following contents:
```bash
NGC_API_KEY="<YOUR_NGC_API_KEY>"
NGC_ORG_ID="<YOUR_NGC_ORG_ID>"
NVIDIA_API_KEY="<YOUR_NVIDIA_AI_FOUNDATION_MODELS_API_KEY>"
```

Note: A file named `default.env` is already provided in the `cyber_dev_day` directory. You can use this file as a template and fill in the necessary values.

## Build Instructions

The following instructions are for building the necessary containers for the Cyber Developer Day 2024.

1. Set the `MORPHEUS_ROOT` environment variable to the root of the Morpheus repository:
   ```bash
   export MORPHEUS_ROOT=$(git rev-parse --show-toplevel)
   ```
2. Ensure all submodules are correctly checked out:
   ```bash
   git submodule update --init --recursive
   ```
3. Pull the LFS files to ensure all Git files are downloaded:
   ```bash
   ${MORPHEUS_ROOT}/scripts/fetch_data.py fetch examples
   ```
4. Build the Morpheus container:
   ```bash
   cd ${MORPHEUS_ROOT}/examples/cyber_dev_day
   docker compose build morpheus
   ```
   This is necessary since we may be using a pre-release version of Morpheus and need the latest changes. This will generate a new container which is only needed by the next build step.
5. Build the Cyber Developer Day containers:
   ```bash
   cd ${MORPHEUS_ROOT}/examples/cyber_dev_day
   docker compose build cyber-dev-day
   ```

## Running the Cyber Developer Day Content

The Cyber Developer Day content is designed to be run using the `docker compose` command. The main entry point is the `cyber-dev-day` container, which is built in the previous step. This container launches a JupyterLab server with the necessary environment variables set to access the NeMo Inference Service and NVIDIA AI Foundation Models API. From there, the pipelines and all content can be run from JupyterLab.

### Launching the Container and Connecting to JupyterLab

To run the Cyber Developer Day content, use the following command:
```bash
cd ${MORPHEUS_ROOT}/examples/cyber_dev_day
docker compose up cyber-dev-day
```

Once launched, you should see a link in the output to connect to the JupyterLab server. Open this link in your web browser to access the content. For example:
```
cyber-dev-day-1  |     To access the server, open this file in a browser:
cyber-dev-day-1  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
cyber-dev-day-1  |     Or copy and paste one of these URLs:
cyber-dev-day-1  |         http://localhost:8888/lab?token=a2d7504f70a2f5407236be5897ee266dc24bf19b01c222bc
cyber-dev-day-1  |         http://127.0.0.1:8888/lab?token=a2d7504f70a2f5407236be5897ee266dc24bf19b01c222bc
```

### Running the Notebook

Once connected to the JupyterLab server, you can navigate to the `notebooks` directory and open the `cyber-dev-day.ipynb` Notebook. The notebook contains the instructions and all of the necessary content to run the Cyber Developer Day.

### Stopping the Container

To stop the container, use the following command:
```bash
cd ${MORPHEUS_ROOT}/examples/cyber_dev_day
docker compose down
```
