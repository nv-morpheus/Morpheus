<!--
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Example cyBERT Morpheus Pipeline for Apache Log Parsing

Example Morpheus pipeline using Docker containers for Triton Inference server and Morpheus SDK/Client.

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```
docker pull nvcr.io/nvidia/tritonserver:22.02-py3
```

##### Start Triton Inference Server container
```
cd ${MORPHEUS_ROOT}/models

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD:/models nvcr.io/nvidia/tritonserver:22.02-py3 tritonserver --model-repository=/models/triton-model-repo --model-control-mode=explicit --load-model log-parsing-onnx
```

### Run Log Parsing Pipeline

Run the following in your Morpheus container to start the log parsing pipeline:

```
python ./examples/log_parsing/run.py \
    --num_threads 1 \
    --input_file ./models/datasets/validation-data/log-parsing-validation-data-input.csv \
    --output_file ./log-parsing-output.jsonlines \
    --model_vocab_hash_file=./data/bert-base-cased-hash.txt \
    --model_vocab_file=./data/bert-base-cased-vocab.txt \
    --model_seq_length=256 \
    --model_name log-parsing-onnx \
    --model_config_file=./models/log-parsing-models/log-parsing-config-20220418.json \
    --server_url localhost:8001
```

Use `--help` to display information about the command line options:

```
python ./examples/log-parsing/run.py --help

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
  --model_vocab_hash_file FILE    Model vocab hash file to use for pre-
                                  processing  [required]
  --model_vocab_file FILE         Model vocab file to use for post-processing
                                  [required]
  --model_seq_length INTEGER RANGE
                                  Sequence length to use for the model  [x>=1]
  --model_name TEXT               The name of the model that is deployed on
                                  Triton server  [required]
  --model_config_file TEXT        Model config file  [required]
  --server_url TEXT               Tritonserver url  [required]
  --help                          Show this message and exit.
```
