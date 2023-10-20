<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

### Launching Triton

Pull the Docker image for Triton:
```bash
docker pull nvcr.io/nvidia/tritonserver:23.06-py3
```

From the Morpheus repo root directory, run the following to launch Triton and load the `all-MiniLM-L6-v2` model:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository=/models/triton-model-repo --exit-on-error=false --model-control-mode=explicit --load-model all-MiniLM-L6-v2
```

This will launch Triton and only load the `all-MiniLM-L6-v2` model. Once Triton has loaded the model, the following will be displayed:
```
+------------------+---------+--------+
| Model            | Version | Status |
+------------------+---------+--------+
| all-MiniLM-L6-v2 | 1       | READY  |
+------------------+---------+--------+
```

## Running the Pipeline

```bash
python examples/llm/main.py vdb_upload pipeline
``````

> **Note**: This pipeline will, by default, run continuously repeatedly polling the configured RSS sources. To run for a fixed number of iterations, add the `--stop_after=N` flag.
