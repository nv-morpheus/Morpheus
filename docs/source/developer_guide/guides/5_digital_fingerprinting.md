<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Digital Fingerprinting (DFP)

## Overview
Every account, user, service, and machine has a digital fingerprint that represents the typical actions performed and not performed over a given period of time. Understanding every entity's day-to-day, moment-by-moment work helps us identify anomalous behavior and uncover potential threats in the environment​.

To construct this digital fingerprint, we will be training unsupervised behavioral models at various granularities, including a generic model for all users in the organization along with fine-grained models for each user to monitor their behavior. These models are continuously updated and retrained over time​, and alerts are triggered when deviations from normality occur for any user​.

## Training Sources
The data we will want to use for the training and inference will be any sensitive system that the user interacts with, such as VPN, authentication and cloud services. The digital fingerprinting example (`examples/digital_fingerprinting/README.md`) included in Morpheus ingests logs from [Azure Active Directory](https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins), and [Duo Authentication](https://duo.com/docs/adminapi).

The location of these logs could be either local to the machine running Morpheus, a shared file system like NFS, or on a remote store such as [Amazon S3](https://aws.amazon.com/s3/).

### Defining a New Data Source
Additional data sources and remote stores can easily be added using the Morpheus API. The key to applying DFP to a new data source is through the process of feature selection. Any data source can be fed into DFP after some preprocessing to get a feature vector per log/data point. In order to build a targeted model for each entity (such as user, service and machine), the chosen data source needs a field that uniquely identifies the entity we're trying to model.

Adding a new source for the DFP pipeline requires defining five critical pieces:
1. The user ID column specified by the `morpheus.config.Config.ae.userid_column_name` configuration attribute. This can be any column which uniquely identifies the user, account or service being fingerprinted. Examples of possible user IDs could be:
   * A username or full name  (for example, `"johndoe"` or `"Jane Doe"`)
   * User's LDAP ID number
   * A user group (for example, `"sales"` or `"engineering"`)
   * Host name of a machine on the network
   * IP address of a client
   * Name of a service (for example, `"DNS"`, `"Customer DB"`, or `"SMTP"`)

1. The timestamp column specified by the `morpheus.config.Config.ae.timestamp_column_name` configuration attribute. The timestamp column needs to be converted to a `datetime` column refer to [`DateTimeColumn`](6_digital_fingerprinting_reference.md#date-time-column-datetimecolumn).
1. The model's features as a list of strings specified by the `morpheus.config.Config.ae.feature_columns` configuration attribute, which should all be available to the pipeline after the [`DFPPreprocessingStage`](6_digital_fingerprinting_reference.md#preprocessing-stage-dfppreprocessingstage).
1. A [`DataFrameInputSchema`](6_digital_fingerprinting_reference.md#dataframe-input-schema-dataframeinputschema) for the [`DFPFileToDataFrameStage`](6_digital_fingerprinting_reference.md#file-to-dataframe-stage-dfpfiletodataframestage) stage.
1. A [`DataFrameInputSchema`](6_digital_fingerprinting_reference.md#dataframe-input-schema-dataframeinputschema) for the [`DFPPreprocessingStage`](6_digital_fingerprinting_reference.md#preprocessing-stage-dfppreprocessingstage).

## Production Deployment Example

This example is designed to illustrate a full-scale, production-ready, DFP deployment in Morpheus. It contains all of the necessary components (such as a model store), to allow multiple Morpheus pipelines to communicate at a scale that can handle the workload of an entire company.

Key Features:
 * Multiple pipelines are specialized to perform either training or inference
 * Uses a model store to allow the training and inference pipelines to communicate
 * Organized into a docker-compose deployment for easy startup
 * Contains a Jupyter notebook service to ease development and debugging
 * Can be deployed to Kubernetes using provided Helm charts
 * Uses many customized stages to maximize performance.

This example is described in `examples/digital_fingerprinting/production/README.md` as well as the rest of this document.

## DFP Features

### Azure Active Directory
| Feature | Description |
| ------- | ----------- |
| `appDisplayName` | for example, `Windows sign in`, `MS Teams`, `Office 365`​ |
| `clientAppUsed` | for example, `IMAP4`, `Browser`​ |
| `deviceDetail.displayName` | for example, `username-LT​` |
| `deviceDetail.browser` | for example, `EDGE 98.0.xyz`, `Chrome 98.0.xyz`​ |
| `deviceDetail.operatingSystem` | for example, `Linux`, `IOS 15`, `Windows 10`​ |
| `statusfailureReason` | for example, `external security challenge not satisfied`, `error validating credentials`​ |
| `riskEventTypesv2` | `AzureADThreatIntel`, `unfamiliarFeatures`​ |
| `location.countryOrRegion` | country or region name​ |
| `location.city` | city name |

#### Derived Features
| Feature | Description |
| ------- | ----------- |
| `logcount` | tracks the number of logs generated by a user within that day (increments with every log)​ |
| `locincrement` | increments every time we observe a new city (`location.city`) in a user's logs within that day​ |
| `appincrement` | increments every time we observe a new app (`appDisplayName`) in a user's logs within that day​ |

### Duo Authentication
| Feature | Description |
| ------- | ----------- |
| `auth_device.name` | phone number​ |
| `access_device.browser` | for example, `Edge`, `Chrome`, `Chrome Mobile​` |
| `access_device.os` | for example, `Android`, `Windows`​ |
| `result` | `SUCCESS` or `FAILURE` ​ |
| `reason` | reason for the results, for example, `User Cancelled`, `User Approved`, `User Mistake`, `No Response`​ |
| `access_device.location.city` | city name |

#### Derived Features
| Feature | Description |
| ------- | ----------- |
| `logcount` | tracks the number of logs generated by a user within that day (increments with every log)​ |
| `locincrement` | increments every time we observe a new city (`location.city`) in a user's logs within that day​ |


## High Level Architecture
DFP in Morpheus is accomplished via two independent pipelines: training and inference​. The pipelines communicate via a shared model store ([MLflow](https://mlflow.org/)), and both share many common components​, as Morpheus is composed of reusable stages that can be easily mixed and matched.

![High Level Architecture](img/dfp_high_level_arch.png)

### Training Pipeline
* Trains user models and uploads to the model store​
* Capable of training individual user models or a fallback generic model for all users​

### Inference Pipeline
* Downloads user models from the model store​
* Generates anomaly scores per log​
* Sends detected anomalies to monitoring services

### Monitoring
* Detected anomalies are published to an S3 bucket, directory or a Kafka topic.
* Output can be integrated with a monitoring tool.

## Runtime Environment Setup
![Runtime Environment Setup](img/dfp_runtime_env.png)

DFP in Morpheus is built as an application of containerized services​ and can be run in two ways:
1. Using docker-compose for testing and development​
1. Using helm charts for production Kubernetes deployment​

### Services
The reference architecture is composed of the following services:​
| Service | Description |
| ------- | ----------- |
| `mlflow` | [MLflow](https://mlflow.org/) provides a versioned model store​ |
| `jupyter` | [Jupyter Server](https://jupyter-server.readthedocs.io/en/latest/)​ necessary for testing and development of the pipelines​ |
| `morpheus_pipeline` | Used for executing both training and inference pipelines |

### Running via `docker-compose`
#### System requirements
* [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/) installed on the host machine​
* Supported GPU with [NVIDIA Container Toolkit​](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

> **Note:**  For GPU Requirements refer to the [Getting Started](../../getting_started.md#requirements) guide.

#### Building the services
From the root of the Morpheus repo, run:
```bash
cd examples/digital_fingerprinting/production
export MORPHEUS_CONTAINER_VERSION="$(git describe --tags --abbrev=0)-runtime"
docker compose build
```

> **Note:** This requires version 1.28.0 or higher of Docker Compose, and preferably v2. If you encounter an error similar to:
>
> ```
> ERROR: The Compose file './docker-compose.yml' is invalid because:
> services.jupyter.deploy.resources.reservations value Additional properties are not allowed ('devices' was
> unexpected)
> ```
>
> This is most likely due to using an older version of the `docker-compose` command, instead re-run the build with `docker compose`. Refer to [Migrate to Compose V2](https://docs.docker.com/compose/migrate/) for more information.

#### Downloading the example datasets
First, we will need to install additional requirements in to the Conda environment. Then run the `examples/digital_fingerprinting/fetch_example_data.py` script. This will download the example data into the `examples/data/dfp` dir.

From the Morpheus repo, run:
```bash
conda env update --solver=libmamba \
  -n ${CONDA_DEFAULT_ENV} \
  --file ./conda/environments/examples_cuda-125_arch-x86_64.yaml

python examples/digital_fingerprinting/fetch_example_data.py all
```

#### Running the services
##### Jupyter Server
From the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose up jupyter
```

Once the build is complete and the service has started, a message similar to the following should display:
```
jupyter  |     To access the server, open this file in a browser:
jupyter  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyter  |     Or copy and paste one of these URLs:
jupyter  |         http://localhost:8888/lab?token=<token>
jupyter  |      or http://127.0.0.1:8888/lab?token=<token>
```

Copy and paste the URL into a web browser. There are four notebooks included with the DFP example:
* dfp_azure_training.ipynb - Training pipeline for Azure Active Directory data
* dfp_azure_inference.ipynb - Inference pipeline for Azure Active Directory data
* dfp_duo_training.ipynb - Training pipeline for Duo Authentication
* dfp_duo_inference.ipynb - Inference pipeline for Duo Authentication

> **Note:** The token in the URL is a one-time use token and a new one is generated with each invocation.

##### Morpheus Pipeline
By default, the `morpheus_pipeline` will run the training pipeline for Duo data from the `examples/digital_fingerprinting/production` dir run:
```bash
docker compose up morpheus_pipeline
```

If instead you want to run a different pipeline from the `examples/digital_fingerprinting/production` dir, run:
```bash
docker compose run morpheus_pipeline bash
```


From the prompt within the `morpheus_pipeline` container, you can run either the `dfp_azure_pipeline.py` or `dfp_duo_pipeline.py` pipeline scripts.
```bash
python dfp_azure_pipeline.py --help
python dfp_duo_pipeline.py --help
```

Both scripts are capable of running either a training or inference pipeline for their respective data sources. The command-line options for both are the same:
| Flag | Type | Description |
| ---- | ---- | ----------- |
| `--train_users` | One of: `all`, `generic`, `individual`, `none` | Indicates whether or not to train per user or a generic model for all users. Selecting `none` runs the inference pipeline. |
| `--skip_user` | TEXT | User IDs to skip. Mutually exclusive with `only_user` |
| `--only_user` | TEXT | Only users specified by this option will be included. Mutually exclusive with `skip_user` |
| `--start_time` | TEXT | The start of the time window, if undefined `start_date` will be `now()-duration` |
| `--duration` | TEXT | The duration to run starting from `start_time` [default: `60d`] |
| `--cache_dir` | TEXT | The location to cache data such as S3 downloads and pre-processed data  [environment variable: `DFP_CACHE_DIR`; default: `./.cache/dfp`] |
| `--log_level` | One of: `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, `DEBUG` | Specify the logging level to use. [default: `WARNING`] |
| `--sample_rate_s` | INTEGER | Minimum time step, in milliseconds, between object logs. [environment variable: `DFP_SAMPLE_RATE_S`; default: 0] |
| `-f`, `--input_file` | TEXT | List of files to process. Can specify multiple arguments for multiple files. Also accepts glob (*) wildcards and schema prefixes such as `s3://`. For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. Refer to [`fsspec` documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files) for list of possible options. |
| `--watch_inputs` | FLAG | Instructs the pipeline to continuously check the paths specified by `--input_file` for new files. This assumes that the at least one paths contains a wildcard. |
| `--watch_interval` | FLOAT | Amount of time, in seconds, to wait between checks for new files. Only used if --watch_inputs is set. [default `1.0`] |
| `--tracking_uri` | TEXT | The MLflow tracking URI to connect to. [default: `http://localhost:5000`] |
| `--help` | | Show this message and exit. |


To run the DFP pipelines with the example datasets within the container, run:

* Duo Training Pipeline
   ```bash
   python dfp_duo_pipeline.py --train_users=all --start_time="2022-08-01" --input_file="/workspace/examples/data/dfp/duo-training-data/*.json"
   ```

* Duo Inference Pipeline
   ```bash
   python dfp_duo_pipeline.py --train_users=none --start_time="2022-08-30" --input_file="/workspace/examples/data/dfp/duo-inference-data/*.json"
   ```

* Azure Training Pipeline
   ```bash
   python dfp_azure_pipeline.py --train_users=all --start_time="2022-08-01" --input_file="/workspace/examples/data/dfp/azure-training-data/*.json"
   ```

* Azure Inference Pipeline
   ```bash
   python dfp_azure_pipeline.py --train_users=none  --start_time="2022-08-30" --input_file="/workspace/examples/data/dfp/azure-inference-data/*.json"
   ```

##### Output Fields
The output files will contain those logs from the input dataset for which an anomaly was detected; this is determined by the z-score in the `mean_abs_z` field. By default, any logs with a z-score of 2.0 or higher are considered anomalous. Refer to [`DFPPostprocessingStage`](6_digital_fingerprinting_reference.md#post-processing-stage-dfppostprocessingstage).

Most of the fields in the output files generated by running the above examples are input fields or derived from input fields. The additional output fields are:
| Field | Type | Description |
| ----- | ---- | ----------- |
| `event_time` | TEXT | ISO 8601 formatted date string, the time the anomaly was detected by Morpheus |
| `model_version` | TEXT | Name and version of the model used to performed the inference, in the form of `<model name>:<version>` |
| `max_abs_z` | FLOAT | Max z-score across all features |
| `mean_abs_z` | FLOAT | Average z-score across all features |

In addition to this, for each input feature the following output fields will exist:
| Field | Type | Description |
| ----- | ---- | ----------- |
| `<feature name>_loss` | FLOAT | The loss |
| `<feature name>_z_loss` | FLOAT | The loss z-score |
| `<feature name>_pred` | FLOAT | The predicted value |

Refer to [DFPInferenceStage](6_digital_fingerprinting_reference.md#inference-stage-dfpinferencestage) for more on these fields.

##### Optional MLflow Service
Starting the `morpheus_pipeline` or the `jupyter` service, will start the `mlflow` service in the background. For debugging purposes, it can be helpful to view the logs of the running MLflow service.

From the `examples/digital_fingerprinting/production` dir, run:
```bash
docker compose up mlflow
```

### Running via Kubernetes​
#### System requirements
* [Kubernetes](https://kubernetes.io/) cluster configured with GPU resources​
* [NVIDIA GPU Operator](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/gpu-operator) installed in the cluster

> **Note:**  For GPU Requirements refer to the [Getting Started](../../getting_started.md#requirements) guide.

## Customizing DFP
For details on customizing the DFP pipeline refer to [Digital Fingerprinting (DFP) Reference](./6_digital_fingerprinting_reference.md).
