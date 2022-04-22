# Morpheus SDK

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI framework and pre-trained AI capabilities that allow them to instantaneously inspect all IP traffic across their data center fabric. The Morpheus developer framework allows teams to build their own optimized pipelines that address cybersecurity and information security use cases. Bringing a new level of security to data centers, Morpheus provides development capabilities around dynamic protection, real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

## Getting Started

### Prerequisites

- Pascal architecture or better
- NVIDIA driver `450.80.02` or higher
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- [Git LFS](https://git-lfs.github.com/)

### Installation

#### Ensure Git LFS is Installed

The large model and data files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). These files will be required for running the training/validation scripts and example pipelines for the Morpheus pre-trained models.

If `Git LFS` is not installed before cloning the repository, the large files will not be pulled. If this is the case, follow the instructions for installing `Git LFS` from [here](https://git-lfs.github.com/), and then run the following command.
```bash
git lfs install
```

#### Clone the Repository

```bash
MORPHEUS_HOME=$(pwd)/morpheus
git clone https://github.com/NVIDIA/Morpheus.git $MORPHEUS_HOME
cd $MORPHEUS_HOME
```

**Note:** If the repository was cloned before `Git LFS` was installed, you can ensure you have downloaded the LFS files with the command:

```bash
git lfs pull
```

#### Pre-built `runtime` Docker image

Pre-built Morpheus Docker images can be downloaded from NGC. The `runtime` image includes pre-installed Morpheus and dependencies:

```bash
docker pull nvcr.io/nvidia/morpheus/morpheus:runtime-22.04-latest
```

**Note:** You must be enrolled in the Morpheus Early Access program to download the Morpheus image.

Run the pre-built `runtime` container:

```bash
DOCKER_IMAGE_TAG=runtime-v0.2-latest ./docker/run_container_release.sh
```

#### Manually build `runtime` Docker image

The Morpheus `runtime` image can also be manually built. This allows you to use a Morpheus build from the development branch or other branch/tag.
To manually build the `runtime` image, run the following from the repo root:

#### Building locally (outside a container)

To build Morpheus outside a container, all the necessary dependencies will need to be installed locally or in a virtual environment. Due to the increased complexity of installing outside of a container, this section has been moved to the `CONTRIBUTING.md`. Please see the "Build in a Conda Environment" section for more information.

Note: Once `morpheus` CLI is installed, shell command completion can be installed with:
```bash
./docker/build_container_release.sh
```
This will create an image named `nvcr.io/nvidia/morpheus/morpheus:latest`.

Run the manually built `runtime` container:

```bash
./docker/run_container_release.sh
```

#### Build from source

Build instructions for developers and contributors can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Running Morpheus

Depending on your configuration, it may be necessary to start additional services that Morpheus will interact with, before launching the pipeline. See the following list of stages that require additional services:

 - `from-kafka`/`to-kafka`
   - Requires a running Kafka cluster
   - See the Quick Launch Kafka section.
 - `inf-triton`
   - Requires a running Triton server
   - See the launching Triton section.

### Quick Launch Kafka Cluster

Launching a full production Kafka cluster is outside the scope of this project. However, if a quick cluster is needed for testing or development, one can be quickly launched via Docker Compose. The following commands outline that process. See [this](https://medium.com/big-data-engineering/hello-kafka-world-the-complete-guide-to-kafka-with-docker-and-python-f788e2588cfc) guide for more in depth information:

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

8. Generate input messages
   1.  In order for Morpheus to read from Kafka, messages need to be published to the cluster. For debugging/testing purposes, the following container can be used:

         ```bash
         # Download from https://netq-shared.s3-us-west-2.amazonaws.com/kafka-producer.tar.gz
         wget https://netq-shared.s3-us-west-2.amazonaws.com/kafka-producer.tar.gz
         # Load container
         docker load --input kafka-producer.tar.gz
         # Run the producer container
         docker run --rm -it -e KAFKA_BROKER_SERVERS=$(broker-list.sh) -e INPUT_FILE_NAME=$MY_INPUT_FILE -e TOPIC_NAME=$MY_INPUT_TOPIC_NAME --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:1
         ```
         In order for this to work, your input file must be accessible from `$PWD`.
   2. You can view the messages with:
         ```bash
         ./start-kafka-shell.sh $KAFKA_ADVERTISED_HOST_NAME
         $KAFKA_HOME/bin/kafka-console-consumer.sh --topic=$MY_TOPIC --bootstrap-server `broker-list.sh`
         ```

### Launching Triton Server

To launch Triton server, use the following command:
```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $PWD/models:/models \
  nvcr.io/nvidia/tritonserver:21.12-py3 \
    tritonserver --model-repository=/models/triton-model-repo \
                 --exit-on-error=false \
                 --model-control-mode=explicit \
                 --load-model abp-nvsmi-xgb \
                 --load-model sid-minibert-onnx \
                 --load-model phishing-bert-onnx
```
This will launch Triton using the port 8001 for the GRPC server. This needs to match the Morpheus configuration.

## Configuration

The Morpheus pipeline can be configured in two ways:
1. Manual configuration in Python script.
2. Configuration via the provided CLI (i.e. `morpheus`)

### Starting the Pipeline (via Manual Python Config)

See the `./examples` directory for examples on how to configure a pipeline via Python.

### Starting the Pipeline (via CLI)

The provided CLI (`morpheus`) is capable of running the included tools as well as any linear pipeline. Instructions for using the CLI can be queried with:
```bash
$ morpheus
Usage: morpheus [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug            [default: no-debug]
  --log_level [CRITICAL|FATAL|ERROR|WARN|WARNING|INFO|DEBUG]
                                  Specify the logging level to use.  [default:
                                  WARNING]
  --log_config_file FILE          Config file to use to configure logging. Use
                                  only for advanced situations. Can accept
                                  both JSON and ini style configurations
  --version                       Show the version and exit.  [default: False]
  --help                          Show this message and exit.  [default:
                                  False]

Commands:
  run    Run one of the available pipelines
  tools  Run a utility tool
```
Each command in the CLI has its own help information. Use `morpheus [command] [...sub-command] --help` to get instructions for each command and sub command. For example:
```bash
$ morpheus run pipeline-nlp inf-triton --help
Configuring Pipeline via CLI
Usage: morpheus run pipeline-nlp inf-triton [OPTIONS]

Options:
  --model_name TEXT               Model name in Triton to send messages to
                                  [required]
  --server_url TEXT               Triton server URL (IP:Port)  [required]
  --force_convert_inputs BOOLEAN  Instructs this stage to forcibly convert all
                                  input types to match what Triton is
                                  expecting. Even if this is set to `False`,
                                  automatic conversion will be done only if
                                  there would be no data loss (i.e. int32 ->
                                  int64).  [default: False]
  --use_shared_memory BOOLEAN     Whether or not to use CUDA Shared IPC Memory
                                  for transferring data to Triton. Using CUDA
                                  IPC reduces network transfer time but
                                  requires that Morpheus and Triton are
                                  located on the same machine  [default:
                                  False]
  --help                          Show this message and exit.  [default:
                                  False]
```

#### CLI Stage Configuration

When configuring a pipeline via the CLI, you start with the command `morpheus run pipeline` and then list the stages in order from start to finish. The order that the commands are placed in will be the order that data flows from start to end. The output of each stage will be linked to the input of the next. For example, to build a simple pipeline that reads from kafka, deserializes messages, serializes them, and then writes to a file, use the following:
```bash
$ morpheus run pipeline-nlp from-kafka --input_topic test_pcap deserialize serialize to-file --filename .tmp/temp_out.json
```
You should see some output similar to:
```log
====Building Pipeline====
Added source: <from-kafka-0; KafkaSourceStage(bootstrap_servers=localhost:9092, input_topic=test_pcap, group_id=custreamz, poll_interval=10millis)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <serialize-2; SerializeStage(include=[], exclude=['^ID$', '^_ts_'], output_type=pandas)>
  └─ morpheus.MultiMessage -> pandas.DataFrame
Added stage: <to-file-3; WriteToFileStage(filename=.tmp/temp_out.json, overwrite=False, file_type=auto)>
  └─ pandas.DataFrame -> pandas.DataFrame
====Building Pipeline Complete!====
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
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  gen-viz       (Deprecated) Write out vizualization data frames
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Serializes messages into a text format
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```
And for the FIL pipeline:
```bash
$ morpheus run pipeline-fil --help
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class     Add detected classifications to each message
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Serializes messages into a text format
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```
And for AE pipeline:
```bash
$ morpheus run pipeline-fil --help
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                               [ARGS]...]...

<Help Paragraph Omitted>

Commands:
  add-class        Add detected classifications to each message
  add-scores       Add probability scores to each message
  buffer           (Deprecated) Buffer results
  delay            (Deprecated) Delay results for a certain duration
  filter           Filter message by a classification threshold
  from-cloudtrail  Load messages from a Cloudtrail directory
  gen-viz          (Deprecated) Write out vizualization data frames
  inf-pytorch      Perform inference with PyTorch
  inf-triton       Perform inference with Triton
  monitor          Display throughput numbers at a specific point in the
                   pipeline
  preprocess       Convert messages to tokens
  serialize        Serializes messages into a text format
  timeseries       Perform time series anomaly detection and add prediction.
  to-file          Write all messages to a file
  to-kafka         Write all messages to a Kafka cluster
  train-ae         Deserialize source data from JSON
  validate         Validates pipeline output against an expected output

```
Note: The available commands for different types of pipelines are not the same. And the same stage in different pipelines may have different options. Please check the CLI help for the most up to date information during development.


## Pipeline Validation

To verify that all pipelines are working correctly, validation scripts have been added at `${MORPHEUS_ROOT}/scripts/validation`. There are scripts for each of the main workflows: Anomalous Behavioral Profilirun_container_release.shng (ABP), Humans-as-Machines-Machines-as-Humans (HAMMAH), Phishing Detection (Phishing), and Sensitive Information Detection (SID).

To run all of the validation workflow scripts, use the following commands:

```bash
# Install utils for checking output
apt update && apt install -y jq bc

# Run validation scripts
./scripts/validation/val-run-all.sh
```

At the end of each workflow, a section will print the different inference workloads that were run and the validation error percentage for each. For example:

```bash
===ERRORS===
PyTorch     :3/314 (0.96 %)
Triton(ONNX):Skipped
Triton(TRT) :Skipped
TensorRT    :Skipped
Complete!
```

This indicates that only 3 out of 314 rows did not match the validation dataset. If you see errors similar to `:/ ( %)` or very high percentages, then the workflow did not complete sucessfully.
