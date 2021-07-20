<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

<!-- TODO: Add License -->

# Morpheus SDK

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI framework and pre-trained AI capabilities that allow them to instantaneously inspect all IP traffic across their data center fabric. The Morpheus developer framework allows teams to build their own optimized pipelines that address cybersecurity and information security use cases. Bringing a new level of security to data centers, Morpheus provides development capabilities around dynamic protection, real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- `conda` or `mamba`
  - Conda is only necessary when building outside of the container.
  - See the [Getting Started Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) if `conda` is not already installed
  - [Optional] Install `mamba` to speed up the package solver:

      ```bash
      conda activate base
      conda install -c conda-forge mamba
      ```

  - **Note:** `mamba` should only be installed once in the base environment

### Installation

#### Pre-built container

Pre-built Morpheus containers can be downloaded from NGC. To download the Morpeus SDK CLI container, run the following command:

```bash
docker pull nvcr.io/ea-nvidia-morpheus/morpheus-sdk-cli:latest
```

**Note:** You must be enrolled in the Morpheus Early Access program to download the Morpheus SDK CLI image.

#### Building locally (inside a container)

To manually build the container, run the following from the repo root:

```bash
docker build -t morpheus -f docker/Dockerfile .
```

This will create the `morpheus:latest` docker image. The commands to run Morpheus in the container or locally are the same. See the [Running Morpheus](#running-morpheus) section for more info.

#### Building locally (outside a container)

To build Morpheus outside of a container, first ensure all of the necessary requirements are installed. The easiest way to ensure the necessary dependencies are installed is to create a new `conda` environment using the following commands:

```bash
export CUDA_VER=11.2
conda env create -n morpheus \
  --file docker/conda/environments/dev_cuda${CUDA_VER}.yml
conda activate morpheus

# Force reinstall of streamz fork
pip install --upgrade --no-deps --force-reinstall git+https://github.com/mdemoret-nv/streamz.git@async#egg=streamz

# Install the NVIDIA pip index
pip install nvidia-pyindex

# ===== OPTIONAL =====
# The following dependencies are optional and only required depending on the workflow and stages that are needed

# TensorRT
TENSORRT_VERSION=7.2.2.3
pip install nvidia-tensorrt==${TENSORRT_VERSION}

# PyTorch (works for CUDA 11.1 & 11.2)
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Then, from the repo root, run the following to build Morpheus and install into the current python environment:

```bash
# For non-developers
pip install .

# For developers (Installs a symlink that allows updating the code)
pip install -e .
```

Additionally, shell command completion can be installed with:
```bash
morpheus tools autocomplete install
```
This will autodetermine your shell and the proper install location. See `morpheus tools autocomplete install --help` for more info on options and available shells.

## Running Morpheus

Depending on your configuration, it may be necessary to start additional services that Morpheus will interact with, before launching the pipeline. See the following list of stages that require additional services:

 - `from-kafka`/`to-kafka`
   - Requires a running Kafka cluster
   - See the Quick Launch Kafka section.
 - `inf-triton`
   - Requires a running Trion server
   - See the launching Triton section.

### Quick Launch Kafka Cluster

Launching a full production Kafka cluster is outside of the scope of this project. However, if a quick cluster is needed for testing or development, one can be quickly launched via Docker Compose. The following commands outline that process. See [this](https://medium.com/big-data-engineering/hello-kafka-world-the-complete-guide-to-kafka-with-docker-and-python-f788e2588cfc) guide for more in depth information:

1. Install `docker-compose` if not already installed:
   ```bash
   conda install -c conda-forge docker-compose
   ```
2. Clone the `kafka-docker` repo from the Morpheus repo root:
   ```bash
   git clone https://github.com/wurstmeister/kafka-docker.git
   ```
3. Change directory to `kafka-docker`
   ```bash
   cd kafka-docker
   ```
4. Export the IP address of your Docker `bridge` network
   ```bash
   export KAFKA_ADVERTISED_HOST_NAME=$(docker network inspect bridge | jq -r '.[0].IPAM.Config[0].Gateway')
   ```
5. Update the `kafka-docker/docker-compose.yml` so the environment variable `KAFKA_ADVERTISED_HOST_NAME` matches the previous step. For example, the line should look like:
   ```yml
   environment:
      KAFKA_ADVERTISED_HOST_NAME: 172.17.0.1
   ```
   Which should match the value of `$KAFKA_ADVERTISED_HOST_NAME` from the previous step:
   ```bash
   $ echo $KAFKA_ADVERTISED_HOST_NAME
   "172.17.0.1"
   ```
6. Launch kafka with 3 instances
   ```bash
   docker-compose up -d --scale kafka=3
   ```
   In practice, 3 instances has been shown to work well. Use as many instances as required. Keep in mind each instance takes about 1 Gb of memory each.
7. Create the topic
   ```bash
   ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
   $KAFKA_HOME/bin/kafka-topics.sh --create --topic=$MY_INPUT_TOPIC_NAME --bootstrap-server `broker-list.sh`
   ```
   Replace `<INPUT_TOPIC_NAME>` with the input name of your choice. If you are using `to-kafka` ensure your output topic is also created.

8. Generage input messages
   1.  In order for Morpheus to read from Kafka, messages need to be published to the cluster. For debugging/testing purposes, the following container can be used:

         ```bash
         # Download from https://netq-shared.s3-us-west-2.amazonaws.com/kafka-producer.tar.gz
         wget https://netq-shared.s3-us-west-2.amazonaws.com/kafka-producer.tar.gz
         # Load container
         docker load --input kafka-producer.tar.gz
         # Run the producer container
         docker run --rm -it -e KAFKA_BROKER_SERVERS=$(broker-list.sh) -e INPUT_FILE_NAME=$MY_INPUT_FILE -e TOPIC_NAME=$MY_INPUT_TOPIC_NAME --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:1
         ```
         In order for this to work, your input file must be accessable from `$PWD`.
   2. You can view the messages with:
         ```bash
         ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
         $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=$MY_TOPIC --bootstrap-server `broker-list.sh`
         ```

### Launching Triton Server

To launch Triton server, use the following command:
```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.02-py3 tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=1
```
This will launch Triton using the port 8001 for the GRPC server. This needs to match the Morpheus configuration.

### Launching Triton Server (With FIL Backend)

If a FIL model is required for the Morpheus pipeline, you will need a version of Triton with the FIL backend. Run the following command to build/launch Triton with a FIL backend.
```bash
# Build FIL Backend for Triton
git clone git@github.com:wphicks/triton_fil_backend.git
cd triton_fil_backend
docker build -t triton_fil -f ops/Dockerfile .

# Run Triton with FIL
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$PWD/triton_models:/models triton_fil:latest tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=1
```

See the Readme in [this](github.com:wphicks/triton_fil_backend) repo for more information.

## Configuration

The Morpheus pipeline can be configured in two ways:
1. Manual configuration in Python script.
2. Configuration via the provided CLI (i.e. `morpheus`)

### Starting the Pipeline (via Manual Python Config)

See the `./examples` directory for examples on how to config via Python. More detailed instructions will be provided in the future.

### Starting the Pipeline (via CLI)

The provided CLI (`morpheus`) is capable of running the included tools as well as any linear pipeline. Instructions for using the CLI can be queried with:
```bash
$ morpheus
Usage: morpheus [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug  [default: False]
  --help                Show this message and exit.  [default: False]

Commands:
  run    Run one of the available pipelines
  tools  Run a utility tool
```
Each command in the CLI has it's own help information. Use `morpheus [command] [...sub-command] --help` to get instructions for each command and sub command. For example:
```bash
$ morpheus run pipeline-nlp inf-triton --help

Usage: morpheus run pipeline-nlp inf-triton [OPTIONS]

Options:
  --model_name TEXT  Model name in Triton to send messages to  [required]
  --server_url TEXT  Triton server URL (IP:Port)  [required]
  --help             Show this message and exit.  [default: False]
```

#### CLI Stage Configuration

When configuring a pipeline via the CLI, you start with the command `morpheus run pipeline` and then list the stages in order from start to finish. The order that the commands are placed in will be the order that data flows from start to end. The output of each stage will be linked to the input of the next. For example, to build a simple pipeline that reads from kafka, deserializes messages, serializes them, and then writes to a file, use the following:
```bash
$ morpheus run pipeline-nlp from-kafka --input_topic test_pcap deserialize serialize to-file --filename .tmp/temp_out.json
```
You should see some output similar to:
```log
====Building Pipeline====
Added source: from-kafka -> <class 'cudf.core.dataframe.DataFrame'>
Added stage: deserialize -> <class 'morpheus.pipeline.messages.MultiMessage'>
Added stage: serialize -> typing.List[str]
Added stage: to-file -> typing.List[str]
====Starting Inference====
```
This is important because it shows you the order of the stages and the output type of each one. Since some stages cannot accept all types of inputs, Morpheus will report an error if you have configured your pipeline incorrectly. For example, if we run the same command as above but forget the `serialize` stage, you will see the following:
```bash
$ morpheus run pipeline-nlp from-kafka --input_topic test_pcap deserialize to-file --filename .tmp/temp_out.json --overwrite

====Building Pipeline====
Added source: from-kafka -> <class 'cudf.core.dataframe.DataFrame'>
Added stage: deserialize -> <class 'morpheus.pipeline.messages.MultiMessage'>

Traceback (most recent call last):
  File "morpheus/pipeline/pipeline.py", line 228, in build_and_start
    current_stream_and_type = await s.build(current_stream_and_type)
  File "morpheus/pipeline/pipeline.py", line 108, in build
    raise RuntimeError("The {} stage cannot handle input of {}. Accepted input types: {}".format(
RuntimeError: The to-file stage cannot handle input of <class 'morpheus.pipeline.messages.MultiMessage'>. Accepted input types: (typing.List[str],)
```
This indicates that the `to-file` stage cannot accept the input type of `morpheus.pipeline.messages.MultiMessage`. This is because the `to-file` stage has no idea how to write that class to a file, it only knows how to write strings. To ensure you have a valid pipeline, look at the `Accepted input types: (typing.List[str],)` portion of the message. This indicates you need a stage that converts from the output type of the `deserialize` stage, `morpheus.pipeline.messages.MultiMessage`, to `typing.List[str]`, which is exactly what the `serialize` stage does.

## Pipeline Stages

A complete list of the pipeline stages will be added in the future. For now, you can query the available stages for each pipeline type via:
```bash
$ morpheus run pipeline-nlp --help
Usage: morpheus run pipeline-nlp [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class     Add detected classifications to each message
  buffer        Buffer results
  delay         Delay results
  deserialize   Deserialize source data from JSON
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  gen-viz       Write out vizualization data frames
  inf-identity  Perform a no-op inference for testing
  inf-triton    Perform inference with Triton
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Deserialize source data from JSON
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
```
And for the FIL pipeline:
```bash
$ morpheus run pipeline-fil --help
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  buffer        Buffer results
  delay         Delay results
  deserialize   Deserialize source data from JSON
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  inf-identity  Perform a no-op inference for testing
  inf-triton    Perform inference with Triton
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Deserialize source data from JSON
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
```

Note: The available commands for different types of pipelines are not the same. And the same stage in different pipelines may have different options. Please check the CLI help for the most up to date information during development.
