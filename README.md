![NVIDIA Morpheus](./img/morpheus-banner.png "Morpheus banner image")

# NVIDIA Morpheus

NVIDIA Morpheus is an open AI application framework that provides cybersecurity developers with a highly optimized AI framework and pre-trained AI capabilities that allow them to instantaneously inspect all IP traffic across their data center fabric. The Morpheus developer framework allows teams to build their own optimized pipelines that address cybersecurity and information security use cases. Bringing a new level of security to data centers, Morpheus provides development capabilities around dynamic protection, real-time telemetry, adaptive policies, and cyber defenses for detecting and remediating cybersecurity threats.

## Documentation
Full documentation (including a quick start guide, a developer/user guide, and API documentation) is available online at [https://docs.nvidia.com/morpheus/](https://docs.nvidia.com/morpheus/).

## Getting Started with Morpheus
There are three ways to get started with Morpheus:
- Using pre-built Docker containers
- Building the Morpheus Docker container
- Building Morpheus from source

The pre-built Docker containers are the easiest way to get started with the latest release of Morpheus. Instructions on how to download and run these containers, including the necessary data and models, can be found on NGC [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/morpheus/collections/morpheus_).

More advanced users, or those who are interested in using the latest pre-release features, will need to build the Morpheus container or build from source. Step-by-step instructions for these users can be found in the following section.

### Prerequisites
The following sections must be followed prior to building the Morpheus container or building Morpheus from source.

#### Requirements
- Pascal architecture GPU or better
- NVIDIA driver `450.80.02` or higher
- [Docker](https://docs.docker.com/get-docker/)
- [The NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- [NVIDIA Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) `22.06` or higher
- [Git LFS](https://git-lfs.github.com/)


#### Clone the Repository

```bash
MORPHEUS_ROOT=$(pwd)/morpheus
git clone https://github.com/NVIDIA/Morpheus.git $MORPHEUS_ROOT
cd $MORPHEUS_ROOT
```

#### Git LFS

The large model and data files in this repo are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/). Only those files which are strictly needed to run Morpheus are downloaded by default when the repository is cloned.

The `scripts/fetch_data.py` script can be used to fetch the Morpheus pre-trained models, and other files required for running the training/validation scripts and example pipelines.

Usage of the script is as follows:
```bash
scripts/fetch_data.py fetch <dataset> [<dataset>...]
```

At time of writing the defined datasets are:
* all - Metaset includes all others
* examples - Data needed by scripts in the `examples` subdir
* models - Morpheus models (largest dataset)
* tests - Data used by unittests
* validation - Subset of the models dataset needed by some unittests

To download just the examples and models:
```bash
scripts/fetch_data.py fetch examples models
```

To download the data needed for unittests:
```bash
scripts/fetch_data.py fetch tests validation
```

If `Git LFS` is not installed the before cloning the repository, the `scripts/fetch_data.py` script will fail. If this is the case follow the instructions for installing `Git LFS` from [here](https://git-lfs.github.com/), and then run the following command:
```bash
git lfs install
```

### Build Morpheus Container

To assist in building the Morpheus container, several scripts have been provided in the `./docker` directory. To build the "release" container, run the following:

```bash
./docker/build_container_release.sh
```

This will create an image named `nvcr.io/nvidia/morpheus/morpheus:${MORPHEUS_VERSION}-runtime` where `$MORPHEUS_VERSION` is replaced by the output of `git describe --tags --abbrev=0`.

To run the built "release" container, use the following:

```bash
./docker/run_container_release.sh
```

You can specify different Docker images and tags by passing the script the `DOCKER_IMAGE_TAG`, and `DOCKER_IMAGE_TAG` variables respectively. For example, to run version `v22.09.00a` use the following:

```bash
DOCKER_IMAGE_TAG="v22.09.00a-runtime" ./docker/run_container_release.sh
```

### Build from Source

It's possible to build from source outside of a container. However, due to the large number of dependencies, this can be complex and is only necessary for developers. Instructions for developers and contributors can be found in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Launching Triton Server

Many of the validation tests and example workflows require a Triton server to function.
Use the following command to launch a Docker container for Triton loading all of the included pre-trained models:

```bash
docker run --rm -ti --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
	-v $PWD/models:/models \
	nvcr.io/nvidia/tritonserver:22.08-py3 \
	tritonserver --model-repository=/models/triton-model-repo \
		--exit-on-error=false \
		--log-info=true \
		--strict-readiness=false
```
This will launch Triton using the default network ports (8000 for HTTP, 8001 for GRPC, and 8002 for metrics).

## Running Morpheus

To run Morpheus, users will need to choose from the Morpheus Command Line Interface (CLI) or Python interface. Which interface to use depends on the user's needs, amount of customization, and operating environment. More information on each interface can be found below.

### Morpheus Python Interface

The Morpheus python interface allows users to configure their pipelines using a python script file. This is ideal for users who are working in a Jupyter notebook, users who need complex initialization logic or users who have customized stages. Documentation on using the Morpheus python interface can be found at [`docs/source/developer_guide/guides.rst`](./docs/source/developer_guide/guides.rst).

For full example pipelines using the python interface, see the `./examples` directory.

### Morpheus Command Line Interface (CLI)

The CLI allows users to completely configure a Morpheus pipeline directly from a terminal. This is ideal for users who do not need customized stages and for users configuring a pipeline in Kubernetes. The Morpheus CLI can be invoked using the `morpheus` command and is capable of running linear pipelines as well as additional tools. Instructions for using the CLI can be queried directly in the terminal using `morpheus --help`:

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

Several examples on using the Morpheus CLI can be found at [`docs/source/basics/examples.rst`](./docs/source/basics/examples.rst).

#### CLI Stage Configuration

When configuring a pipeline via the CLI, you start with the command `morpheus run pipeline` and then list the stages in order from start to finish. The order that the commands are placed in will be the order that data flows from start to end. The output of each stage will be linked to the input of the next. For example, to build a simple pipeline that reads from Kafka, deserializes messages, serializes them, and then writes to a file, use the following:

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

This indicates that the `to-file` stage cannot accept the input type of `morpheus.pipeline.messages.MultiMessage`. This is because the `to-file` stage has no idea how to write that class to a file; it only knows how to write strings. To ensure you have a valid pipeline, look at the `Accepted input types: (typing.List[str],)` portion of the message. This indicates you need a stage that converts from the output type of the `deserialize` stage, `morpheus.pipeline.messages.MultiMessage`, to `typing.List[str]`, which is exactly what the `serialize` stage does.

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

And for the AE pipeline:

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
Note: The available commands for different types of pipelines are not the same. This means that the same stage, when used in different pipelines, may have different options. Please check the CLI help for the most up-to-date information during development.

## Contributing
Please see our [guide for contributing to Morpheus](./CONTRIBUTING.md).
