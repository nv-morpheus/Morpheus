<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ABP Detection Example Using Morpheus


## Setup
To run this example, an instance of Triton Inference Server and a sample dataset is required. The following steps will outline how to build and run Trtion with the provided FIL model.

### Triton Inference Server
```bash
docker pull nvcr.io/nvidia/tritonserver:22.02-py3
```

##### Deploy Triton Inference Server

Bind the provided `abp-pcap-xgb` directory to the docker container model repo at `/models`.

```bash
# Change directory to the anomalous behavior profiling example folder
cd <MORPHEUS_ROOT>/examples/abp_pcap_detection

# Launch the container
docker run --rm --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/abp-pcap-xgb:/models/abp-pcap-xgb --name tritonserver nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models --exit-on-error=false --model-control-mode=poll --repository-poll-secs=30
```

##### Verify Model Deployment
Once Triton server finishes starting up, it will display the status of all loaded models. Successful deployment of the model will show the following:

```bash
+-----------------------------+---------+--------+
| Model                       | Version | Status |
+-----------------------------+---------+--------+
| abp-pcap-xgb                | 1       | READY  |
+-----------------------------+---------+--------+
```

## ABP Detection Pipeline
Use Morpheus to run the Anomalous Behavior Profiling Detection Pipeline with the pcap data. A pipeline has been configured in `run.py` with several command line options:

```bash
$ python run.py --help
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model  [x>=1]
  --input_file PATH               Input filepath  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model  [x>=1]
  --model_name TEXT               The name of the model that is deployed on
                                  Tritonserver
  --iterative                     Iterative mode will emit dataframes one at a
                                  time. Otherwise a list of dataframes is
                                  emitted. Iterative mode is good for
                                  interleaving source stages.
  --server_url TEXT               Tritonserver url  [required]
  --file_type [auto|json|csv]     Indicates what type of file to read.
                                  Specifying 'auto' will determine the file
                                  type from the extension.
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided at `<MORPHEUS_ROOT>/examples/data`, run the following:

```bash
python run.py \
	--input_file ../data/abp_pcap_dump.jsonlines \
	--output_file ./pcap_out.jsonlines \
	--model_name 'abp-pcap-xgb' \
	--server_url localhost:8001
```
Note: Both Morpheus and Trinton Inference Server containers must have access to the same GPUs in order for this example to work.

The pipeline will process the input `pcap_dump.jsonlines` sample data and write it to `pcap_out.jsonlines`.
