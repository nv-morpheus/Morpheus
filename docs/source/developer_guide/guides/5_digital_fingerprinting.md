<!--
SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# 5. Digital Fingerprinting (DFP)

## Overview
Every account, user, service and machine has a digital fingerprint​, which represents the typical actions performed and not performed over a given period of time.  Understanding every entity's day-to-day, moment-by-moment work helps us identify anomalous behavior and uncover potential threats in the environment​.

To construct this digital fingerprint we will be training unsupervised behavioral models at various granularities, including a generic model for all users in the organization along with fine-grained models for each user to monitor their behavior. These models are continuously updated and retrained over time​, and alerts are triggered when deviations from normality occur for any user​.

## Training Sources
The data we will want to use for the training and inference will be any sensitive system that the user interacts with, such as VPN, authentication and cloud services. The [digital fingerprinting example](/examples/digital_fingerprinting/README.md) included in Morpheus ingests logs from [AWS CloudTrail](https://docs.aws.amazon.com/cloudtrail/index.html), [Azure Active Directory](https://docs.microsoft.com/en-us/azure/active-directory/reports-monitoring/concept-sign-ins) and [Duo Authentication](https://duo.com/docs/adminapi#authentication-logs).

The location of these logs could be either local to the machine running Morpheus, a shared file system like NFS or on a remote store such as [Amazon S3](https://aws.amazon.com/s3/).

### Defining a New Data Source
Additional data sources and remote stores can easily be added using the Morpheus SDK.  The key to applying DFP to a new data source is through the process of feature selection. Any data source can be fed into DFP after some preprocessing to get a feature vector per log/data point​.  In order to build a targeted model for each entity (user/service/machine... etc.), the chosen data source needs a field that uniquely identifies the entity we’re trying to model.

Adding a new source for the DFP pipeline requires defining five critical pieces:
1. The user_id column in the Morpheus config attribute `ae.userid_column_name`. This can be any column which uniquely identifies the user, account or service being fingerprinted. Examples of possible user_ids could be:
   * A username or fullname  (ie. "johndoe", "Jane Doe")
   * User's LDAP ID number
   * A user group (ie. "sales", "engineering")
   * Hostname of a machine on the network
   * IP address of a client
   * Name of a service (ie. "DNS", "Customer DB", "SMTP")

1. The timestamp column in the Morpheus config attribute `ae.timestamp_column_name` and ensure it is converted to a datetime column see [`DateTimeColumn`](#date-time-column-datetimecolumn).
1. The model's features as a list of strings in the Morpheus config attribute `ae.feature_columns` which should all be available to the pipeline after the [`DFPPreprocessingStage`](#preprocessing-stage-dfppreprocessingstage).
1. A [`DataFrameInputSchema`](#dataframe-input-schema-dataframeinputschema) for the [`DFPFileToDataFrameStage`](#file-to-dataframe-stage-dfpfiletodataframestage) stage.
1. A [`DataFrameInputSchema`](#dataframe-input-schema-dataframeinputschema) for the [`DFPPreprocessingStage`](#preprocessing-stage-dfppreprocessingstage).

## DFP Examples
The DFP workflow is provided as two separate examples: a simple, "starter" pipeline for new users and a complex, "production" pipeline for full scale deployments. While these two examples both perform the same general tasks, they do so in very different ways. The following is a breakdown of the differences between the two examples.

### The "Starter" Example

This example is designed to simplify the number of stages and components and provide a fully contained workflow in a single pipeline.

Key Differences:
 * A single pipeline which performs both training and inference
 * Requires no external services
 * Can be run from the Morpheus CLI

This example is described in more detail in [`examples/digital_fingerprinting/starter/README.md`](/examples/digital_fingerprinting/starter/README.md)

### The "Production" Example

This example is designed to show what a full scale, production ready, DFP deployment in Morpheus would look like. It contains all of the necessary components (such as a model store), to allow multiple Morpheus pipelines to communicate at a scale that can handle the workload of an entire company.

Key Differences:
 * Multiple pipelines are specialized to perform either training or inference
 * Requires setting up a model store to allow the training and inference pipelines to communicate
 * Organized into a docker-compose deployment for easy startup
 * Contains a Jupyter notebook service to ease development and debugging
 * Can be deployed to Kubernetes using provided Helm charts
 * Uses many customized stages to maximize performance.

This example is described in [`examples/digital_fingerprinting/production/README.md`](/examples/digital_fingerprinting/production/README.md) as well as the rest of this document.

### DFP Features

#### AWS CloudTrail
| Feature | Description |
| ------- | ----------- |
| userIdentityaccessKeyId | e.g., ACPOSBUM5JG5BOW7B2TR, ABTHWOIIC0L5POZJM2FF, AYI2CM8JC3NCFM4VMMB4 |
| userAgent | e.g., Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 10.0; Trident/5.1), Mozilla/5.0 (Linux; Android 4.3.1) AppleWebKit/536.1 (KHTML, like Gecko) Chrome/62.0.822.0 Safari/536.1, Mozilla/5.0 (Macintosh; U; PPC Mac OS X 10 7_0; rv:1.9.4.20) Gecko/2012-06-10 12:09:43 Firefox/3.8 |
| userIdentitysessionContextsessionIssueruserName | e.g., role-g |
| sourceIPAddress | e.g., 208.49.113.40, 123.79.131.26, 128.170.173.123 |
| userIdentityaccountId | e.g., Account-123456789 |
| errorMessage | e.g., The input fails to satisfy the constraints specified by an AWS service., The specified subnet cannot be found in the VPN with which the Client VPN endpoint is associated., Your account is currently blocked. Contact aws-verification@amazon.com if you have questions. |
| userIdentitytype | e.g., FederatedUser |
| eventName | e.g., GetSendQuota, ListTagsForResource, DescribeManagedPrefixLists |
| userIdentityprincipalId | e.g., 39c71b3a-ad54-4c28-916b-3da010b92564, 0baf594e-28c1-46cf-b261-f60b4c4790d1, 7f8a985f-df3b-4c5c-92c0-e8bffd68abbf |
| errorCode | e.g., success, MissingAction, ValidationError |
| eventSource | e.g., lopez-byrd.info, robinson.com, lin.com |
| userIdentityarn | e.g., arn:aws:4a40df8e-c56a-4e6c-acff-f24eebbc4512, arn:aws:573fd2d9-4345-487a-9673-87de888e4e10, arn:aws:c8c23266-13bb-4d89-bce9-a6eef8989214 |
| apiVersion | e.g., 1984-11-26, 1990-05-27, 2001-06-09 |

#### Azure Active Directory
| Feature | Description |
| ------- | ----------- |
| appDisplayName | e.g., Windows sign in, MS Teams, Office 365​ |
| clientAppUsed | e.g., IMAP4, Browser​ |
| deviceDetail.displayName | e.g., username-LT​ |
| deviceDetail.browser | e.g., EDGE 98.0.xyz, Chrome 98.0.xyz​ |
| deviceDetail.operatingSystem | e.g., Linux, IOS 15, Windows 10​ |
| statusfailureReason | e.g., external security challenge not satisfied, error validating credentials​ |
| riskEventTypesv2 | AzureADThreatIntel, unfamiliarFeatures​ |
| location.countryOrRegion | country or region name​ |
| location.city | city name |

##### Derived Features
| Feature | Description |
| ------- | ----------- |
| logcount | tracks the number of logs generated by a user within that day (increments with every log)​ |
| locincrement | increments every time we observe a new city (location.city) in a user’s logs within that day​ |
| appincrement | increments every time we observe a new app (appDisplayName) in a user’s logs within that day​ |

#### Duo Authentication
| Feature | Description |
| ------- | ----------- |
| auth_device.name | phone number​ |
| access_device.browser | e.g., Edge, Chrome, Chrome Mobile​ |
| access_device.os | e.g., Android, Windows​ |
| result | SUCCESS or FAILURE ​ |
| reason | reason for the results, e.g., User Cancelled, User Approved, User Mistake, No Response​ |
| access_device.location.city | city name |

##### Derived Features
| Feature | Description |
| ------- | ----------- |
| logcount | tracks the number of logs generated by a user within that day (increments with every log)​ |
| locincrement | increments every time we observe a new city (location.city) in a user’s logs within that day​ |


## High Level Architecture
DFP in Morpheus is accomplished via two independent pipelines: training and inference​. The pipelines communicate via a shared model store ([MLflow](https://mlflow.org/)), and both share many common components​, as Morpheus is composed of reusable stages which can be easily mixed and matched.

![High Level Architecture](img/dfp_high_level_arch.png)

#### Training Pipeline
* Trains user models and uploads to the model store​
* Capable of training individual user models or a fallback generic model for all users​

#### Inference Pipeline
* Downloads user models from the model store​
* Generates anomaly scores per log​
* Sends detected anomalies to monitoring services

#### Monitoring
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
| mlflow | [MLflow](https://mlflow.org/) provides a versioned model store​ |
| jupyter | [Jupyter Server](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)​ necessary for testing and development of the pipelines​ |
| morpheus_pipeline | Used for executing both training and inference pipelines |

### Running via `docker-compose`
#### System requirements
* [Docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/) installed on the host machine​
* Supported GPU with [nvidia-docker runtime​](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

> **Note:**  For GPU Requirements see [README.md](/README.md#requirements)

#### Building the services
From the root of the Morpheus repo run:
```bash
cd examples/digital_fingerprinting/production
docker-compose build
```

#### Running the services
##### Jupyter Server
From the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up jupyter
```

Once the build is complete and the service has started you will be prompted with a message that should look something like:
```
jupyter  |     To access the server, open this file in a browser:
jupyter  |         file:///root/.local/share/jupyter/runtime/jpserver-7-open.html
jupyter  |     Or copy and paste one of these URLs:
jupyter  |         http://localhost:8888/lab?token=<token>
jupyter  |      or http://127.0.0.1:8888/lab?token=<token>
```

Copy and paste the url into a web browser. There are four notebooks included with the DFP example:
* dfp_azure_training.ipynb - Training pipeline for Azure Active Directory data
* dfp_azure_inference.ipynb - Inference pipeline for Azure Active Directory data
* dfp_duo_training.ipynb - Training pipeline for Duo Authentication
* dfp_duo_inference.ipynb - Inference pipeline for Duo Authentication

> **Note:** The token in the url is a one-time use token, and a new one is generated with each invocation.

##### Morpheus Pipeline
By default the `morpheus_pipeline` will run the training pipeline for Duo data, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up morpheus_pipeline
```

If instead you wish to run a different pipeline, from the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose run morpheus_pipeline bash
```

From the prompt within the `morpheus_pipeline` container you can run either the `dfp_azure_pipeline.py` or `dfp_duo_pipeline.py` pipeline scripts.
```bash
python dfp_azure_pipeline.py --help
python dfp_duo_pipeline.py --help
```

Both scripts are capable of running either a training or inference pipeline for their respective data sources. The command line options for both are the same:
| Flag | Type | Description |
| ---- | ---- | ----------- |
| `--train_users` | One of: `all`, `generic`, `individual`, `none` | Indicates whether or not to train per user or a generic model for all users. Selecting `none` runs the inference pipeline. |
| `--skip_user` | TEXT | User IDs to skip. Mutually exclusive with `only_user` |
| `--only_user` | TEXT | Only users specified by this option will be included. Mutually exclusive with `skip_user` |
| `--start_time` | TEXT | The start of the time window, if undefined start_date will be `now()-duration` |
| `--duration` | TEXT | The duration to run starting from `start_time` [default: 60d] |
| `--cache_dir` | TEXT | The location to cache data such as S3 downloads and pre-processed data  [env var: `DFP_CACHE_DIR`; default: `./.cache/dfp`] |
| `--log_level` | One of: `CRITICAL`, `FATAL`, `ERROR`, `WARN`, `WARNING`, `INFO`, `DEBUG` | Specify the logging level to use.  [default: `WARNING`] |
| `--sample_rate_s` | INTEGER | Minimum time step, in milliseconds, between object logs.  [env var: `DFP_SAMPLE_RATE_S`; default: 0] |
| `-f`, `--input_file` | TEXT | List of files to process. Can specify multiple arguments for multiple files. Also accepts glob (*) wildcards and schema prefixes such as `s3://`. For example, to make a local cache of an s3 bucket, use `filecache::s3://mybucket/*`. See [fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files) for list of possible options. |
| `--tracking_uri` | TEXT | The MLflow tracking URI to connect to the tracking backend. [default: `http://localhost:5000`] |
| `--help` | | Show this message and exit. |


##### Optional MLflow Service
Starting either the `morpheus_pipeline` or the `jupyter` service, will start the `mlflow` service in the background.  For debugging purposes it can be helpful to view the logs of the running MLflow service.

From the `examples/digital_fingerprinting/production` dir run:
```bash
docker-compose up mlflow
```

### Running via Kubernetes​
#### System requirements
* [Kubernetes](https://kubernetes.io/) cluster configured with GPU resources​
* [NVIDIA GPU Operator](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/gpu-operator) installed in the cluster

> **Note:**  For GPU Requirements see [README.md](/README.md#requirements)

## Morpheus Configuration
![Morpheus Configuration](img/dfp_deployment_configs.png)

### Pipeline Structure Configuration
![Pipeline Structure Configuration](img/dfp_pipeline_structure.png)

The stages in both the Training and Inference pipelines can be mixed and matched with little impact​, i.e., the `MultiFileSource` can be configured to pull from S3 or from local files and can be replaced altogether with any other Morpheus input stage.  Similarly the S3 writer can be replaced with any Morpheus output stage.  Regardless of the inputs & outputs the core pipeline should remain unchanged.  While stages in the core of the pipeline (inside the blue areas in the above diagram) perform common actions that should be configured not exchanged.

### Morpheus Config

For both inference and training pipeline the Morpheus config object should be constructed with the same values, and should look like:
```python
import os

from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import CppConfig
from morpheus.cli.utils import get_package_relative_file
from morpheus.cli.utils import load_labels_file
```
```python
CppConfig.set_should_use_cpp(False)

config = Config()
config.num_threads = os.cpu_count()
config.ae = ConfigAutoEncoder()
config.ae.feature_columns = load_labels_file(get_package_relative_file("data/columns_ae_azure.txt"))
```

Other attributes which might be needed:
| Attribute | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `Config.ae.userid_column_name` | `str` | `userIdentityaccountId` | Column in the `DataFrame` containing the username or user ID |
| `Config.ae.timestamp_column_name` |  `str` | `timestamp` | Column in the `DataFrame` containing the timestamp of the event |
| `Config.ae.fallback_username` | `str` | `generic_user` | Name to use for the generic user model, should not match the name of any real users |

### Schema Definition

#### DataFrame Input Schema (`DataFrameInputSchema`)
The `DataFrameInputSchema` ([examples/digital_fingerprinting/production/morpheus/dfp/utils/column_info.py](/examples/digital_fingerprinting/production/morpheus/dfp/utils/column_info.py)) class defines the schema specifying the columns to be included in the output `DataFrame`.  Within the DFP pipeline there are two stages where pre-processing is performed, the `DFPFileToDataFrameStage` stage and the `DFPPreprocessingStage`.  This decoupling of the pre-processing stages from the actual operations needed to be performed allows for the actual schema to be user-defined in the pipeline and re-usability of the stages.  It is up to the user to define the fields which will appear in the `DataFrame`.  Any column in the input data that isn't specified in either `column_info` or `preserve_columns` constructor arguments will not appear in the output.   The exception to this are JSON fields, specified in the `json_columns` argument which defines json fields which are to be normalized.

It is important to note that the fields defined in `json_columns` are normalized prior to the processing of the fields in `column_info`, allowing for processing to be performed on fields nested in json columns.  For example, say we had a JSON field named `event` containing a key named `timestamp` which in the JSON data appears as an ISO 8601 formatted date string, we could ensure it was converted to a datetime object to downstream stages with the following:
```python
from dfp.utils.column_info import DataFrameInputSchema
from dfp.utils.column_info import DateTimeColumn
```
```python
schema = DataFrameInputSchema(
    json_columns=['event'],
    column_info=[DateTimeColumn(name=config.ae.timestamp_column_name, dtype=datetime, input_name='event.timestamp')])
```

In the above examples three operations were performed:
1. The `event` JSON field was normalized, resulting in new fields prefixed with `event.` to be included in the output `DataFrame`.
1. The newly created field `event.timestamp` is parsed into a datetime field.
1. Since the DFP pipeline explicitly requires a timestamp field, we name this new column with the `config.ae.timestamp_column_name` config attribute ensuring it matches the pipeline configuration. When `name` and `input_name` are the same the old field is overwritten, and when they differ a new field is created.

The `DFPFileToDataFrameStage` is executed first and is responsible for flattening potentially nested JSON data and performing any sort of data type conversions. The `DFPPreprocessingStage` is executed later after the `DFPSplitUsersStage` allowing for the possibility of per-user computed fields such as the `logcount` and `locincrement` fields mentioned previously. Both stages are performed after the `DFPFileBatcherStage` allowing for per time period (per-day by default) computed fields.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `json_columns` | `List[str]` | Optional list of json columns in the incoming `DataFrame` to be normalized (currently using the [`pandas.json_normalize`](https://pandas.pydata.org/docs/reference/api/pandas.json_normalize.html) method). Each key in a json field will be flattened into a new column named `<field>.<key>` for example a json field named `user` containing a key named `id` will result in a new column named `user.id`. By default this is an empty `list`. |
| `column_info` | `List[str]` | Optional list of `ColumnInfo` instances, each defining a specific operation to be performed upon a column. These include renames, type conversions, and custom computations. By default this is an empty `list`. |
| `preserve_columns` | `List[str]` or `str` | Optional regular expression string or list of regular expression strings that define columns in the input data which should be preserved in the output `DataFrame`. By default this is an empty `list`. |
| `row_filter` | `function` or `None` | Optional function to be called after all other processing has been performed. This function receives the `DataFrame` as its only argument returning a `DataFrame`. |

#### Column Info (`ColumnInfo`)
Defines a single column and type-cast.
| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |

#### Custom Column (`CustomColumn`)
Subclass of `ColumnInfo`, defines a column to be computed by a user-defined function `process_column_fn`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `process_column_fn` | `function` | Function which receives the entire `DataFrame` as its only input, returning a new [`pandas.Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) object to be stored in column `name`. |

#### Rename Column (`RenameColumn`)
Subclass of `ColumnInfo`, adds the ability to also perform a rename.
| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |

#### Boolean Column (`BoolColumn`)
Subclass of `RenameColumn`, adds the ability to map a set custom values as boolean values. For example say we had a string input field containing one of 5 possible enum values: `OK`, `SUCCESS`, `DENIED`, `CANCELED` and `EXPIRED` we could map these values into a single boolean field as:
```python
from dfp.utils.column_info import BoolColumn
```
```python
field = BoolColumn(name="result",
                   dtype=bool,
                   input_name="result",
                   true_values=["OK", "SUCCESS"],
                   false_values=["DENIED", "CANCELED", "EXPIRED"])
```

We used strings in this example, however we also could have just as easily mapped integer status codes.  We also have the ability to map onto types other than boolean by providing custom values for true and false  (eg. `1`/`0`, `yes`/`no`) .

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Typically this should be `bool` however it could potentially be another type if `true_value` and `false_value` are specified. |
| `input_name` | `str` | Original column name |
| `true_value` | Any | Optional value to store for true values, should be of a type `dtype`. Defaults to `True`. |
| `false_value` | Any | Optional value to store for false values, should be of a type `dtype`. Defaults to `False`. |
| `true_values` | `List[str]` | List of string values to be interpreted as true. |
| `false_values` | `List[str]` | List of string values to be interpreted as false. |

#### Date-Time Column (`DateTimeColumn`)
Subclass of `RenameColumn`, specific to casting UTC localized datetime values. When incoming values contain a time-zone offset string the values are converted to UTC, while values without a time-zone are assumed to be UTC.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |

#### String-Join Column (`StringJoinColumn`)
Subclass of `RenameColumn`, converts incoming `list` values to string by joining by `sep`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_name` | `str` | Original column name |
| `sep` | `str` | Separator string to use for the join |

#### String-Cat Column (`StringCatColumn`)
Subclass of `ColumnInfo`, concatenates values from multiple columns into a new string column separated by `sep`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Any type string or Python class recognized by [Pandas](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes) |
| `input_columns` | `List[str]` | List of columns to concatenate |
| `sep` | `str` | Separator string |

#### Increment Column (`IncrementColumn`)
Subclass of `DateTimeColumn`, counts the unique occurrences of a value in `groupby_column` over a specific time window `period` based on dates in the `input_name` field.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `name` | `str` | Name of the destination column |
| `dtype` | `str` or Python type | Should be `int` or other integer class |
| `input_name` | `str` | Original column name containing timestamp values |
| `groupby_column` | `str` | Column name to group by |
| `period` | `str` | Optional time period to perform the calculation over, value must be [one of pandas' offset strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases). Defaults to `D` one day |

### Input Stages
![Input Stages](img/dfp_input_config.png)

#### Source Stage (`MultiFileSource`)
The `MultiFileSource`([`examples/digital_fingerprinting/production/morpheus/dfp/stages/multi_file_source.py`](/examples/digital_fingerprinting/production/morpheus/dfp/stages/multi_file_source.py)) receives a path or list of paths (`filenames`), and will collectively be emitted into the pipeline as an [fsspec.core.OpenFiles](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFiles) object.  The paths may include wildcards `*` as well as urls (ex: `s3://path`) to remote storage providers such as S3, FTP, GCP, Azure, Databricks and others as defined by [fsspec](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open_files#fsspec.open_files).  In addition to this paths can be cached locally by prefixing them with `filecache::` (ex: `filecache::s3://bucket-name/key-name`).

> **Note:**  this stage does not actually download the data files, allowing the file list to be filtered and batched prior to being downloaded.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `filenames` | `List[str]` or `str` | Paths to source file to be read from |


#### File Batcher Stage (`DFPFileBatcherStage`)
The `DFPFileBatcherStage` ([`examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_file_batcher_stage.py`](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_file_batcher_stage.py)) groups data in the incoming `DataFrame` in batches of a time period (per day default), and optionally filtering incoming data to a specific time window.  This stage can potentially improve performance by combining multiple small files into a single batch.  This stage assumes that the date of the logs can be easily inferred such as encoding the creation time in the file name (e.g., `AUTH_LOG-2022-08-21T22.05.23Z.json`), or using the modification time as reported by the file system. The actual method for extracting the date is encoded in a user-supplied `date_conversion_func` function (more on this later).

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `date_conversion_func` | `function` | Function receives a single [fsspec.core.OpenFile](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFile) argument and returns a `datetime.datetime` object |
| `period` | `str` | Time period to group data by, value must be [one of pandas' offset strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases) |
| `sampling_rate_s` | `int` | Optional, default=`0`. When non-zero a subset of the incoming data files will be sampled, only including a file if the `datetime` returned by `date_conversion_func` is at least `sampling_rate_s` seconds greater than  the `datetime` of the previously included file |
| `start_time` | `datetime` | Optional, default=`None`. When not None incoming data files will be filtered, excluding any files created prior to `start_time` |
| `end_time` | `datetime` | Optional, default=`None`. When not None incoming data files will be filtered, excluding any files created after `end_time` |

For situations where the creation date of the log file is encoded in the filename, the `date_extractor` in the [`examples/digital_fingerprinting/production/morpheus/dfp/utils/file_utils.py`](/examples/digital_fingerprinting/production/morpheus/dfp/utils/file_utils.py) module can be used.  The `date_extractor` assumes that the timestamps are localized to UTC and will need to have a regex pattern bound to it before being passed in as a parameter to `DFPFileBatcherStage`. The regex pattern will need to contain the following named groups: `year`, `month`, `day`, `hour`, `minute`, `second`, and optionally `microsecond`.  In cases where the regular expression does not match the `date_extractor` function will fallback to using the modified time of the file.

For input files containing an ISO 8601 formatted date string the `iso_date_regex` regex can be used ex:
```python
from functools import partial

from dfp.utils.file_utils import date_extractor
from dfp.utils.file_utils import iso_date_regex
```
```python
# Batch files into buckets by time. Use the default ISO date extractor from the filename
pipeline.add_stage(
    DFPFileBatcherStage(config,
                        period="D",
                        date_conversion_func=functools.partial(date_extractor, filename_regex=iso_date_regex)))
```

> **Note:**  If `date_conversion_func` returns time-zone aware timestamps, then `start_time` and `end_time` if not-None need to also be timezone aware datetime objects.

#### File to DataFrame Stage (`DFPFileToDataFrameStage`)
The `DFPFileToDataFrameStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_file_to_df.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_file_to_df.py)) stage receives a `list` of an [fsspec.core.OpenFiles](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.OpenFiles) and loads them into a single `DataFrame` which is then emitted into the pipeline.  When the parent stage is `DFPFileBatcherStage` each batch (typically one day) is concatenated into a single `DataFrame`.  If the parent was `MultiFileSource` the entire dataset is loaded into a single `DataFrame`.  Because of this, it is important to choose a `period` argument for `DFPFileBatcherStage` small enough such that each batch can fit into memory.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `schema` | `DataFrameInputSchema` | Schema specifying columns to load, along with any necessary renames and data type conversions  |
| `filter_null` | `bool` | Optional: Whether to filter null rows after loading, by default True. |
| `file_type` | `morpheus._lib.file_types.FileTypes` (enum) | Optional: Indicates file type to be loaded. Currently supported values at time of writing are: `FileTypes.Auto`, `FileTypes.CSV`, and `FileTypes.JSON`. Default value is `FileTypes.Auto` which will infer the type based on the file extension, set this value if using a custom extension |
| `parser_kwargs` | `dict` or `None` | Optional: additional keyword arguments to be passed into the `DataFrame` parser, currently this is going to be either [`pandas.read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) or [`pandas.read_json`](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html) |
| `cache_dir` | `str` | Optional: path to cache location, defaults to `./.cache/dfp` |

This stage is able to download and load data files concurrently by multiple methods.  Currently supported methods are: `single_thread`, `multiprocess`, `dask`, and `dask_thread`.  The method used is chosen by setting the `FILE_DOWNLOAD_TYPE` environment variable, and `dask_thread` is used by default, and `single_thread` effectively disables concurrent loading.

This stage will cache the resulting `DataFrame` in `cache_dir`, since we are caching the `DataFrame`s and not the source files, a cache hit avoids the cost of parsing the incoming data. In the case of remote storage systems such as S3 this avoids both parsing and a download on a cache hit.  One consequence of this is that any change to the `schema` will require purging cached files in the `cache_dir` before those changes are visible.

> **Note:**  this caching is in addition to any caching which may have occurred when using the optional `filecache::` prefix.

### Output Stages
![Output Stages](img/dfp_output_config.png)

For the inference pipeline any Morpheus output stage such as `morpheus.stages.output.write_to_file_stage.WriteToFileStage` and `morpheus.stages.output.write_to_kafka_stage.WriteToKafkaStage` could be used in addition to the `WriteToS3Stage` documented below.

#### Write to File Stage (`WriteToFileStage`)
This final stage will write all received messages to a single output file in either CSV or JSON format.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `filename` | `str` | The file to write anomalous log messages to. |
| `overwrite` | `bool` | Optional, defaults to `False`. If the file specified in `filename` already exists, it will be overwritten if this option is set to `True` |

#### Write to S3 Stage (`WriteToS3Stage`)
The `WriteToS3Stage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/write_to_s3_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/write_to_s3_stage.py)) stage writes the resulting anomaly detections to S3.  The `WriteToS3Stage` decouples the S3 specific operations from the Morpheus stage, and as such receives an `s3_writer` argument.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `s3_writer` | `function` | User defined function which receives an instance of a `morpheus.messages.message_meta.MessageMeta` and returns that same message instance. Any S3 specific configurations such as bucket name should be bound to the method. |

### Core Pipeline
These stages are common to both the training & inference pipelines, unlike the input & output stages these are specific to the DFP pipeline and intended to be configured but not replaceable.

#### Split Users Stage (`DFPSplitUsersStage`)
The `DFPSplitUsersStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_split_users_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_split_users_stage.py)) stage receives an incoming `DataFrame` and emits a `list` of `DFPMessageMeta` where each `DFPMessageMeta` represents the records associated for a given user.  This allows for downstream stages to perform all necessary operations on a per user basis.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `include_generic` | `bool` | When `True` a `DFPMessageMeta` will be constructed for the generic user containing all records not excluded by the `skip_users` and `only_users` filters |
| `include_individual` | `bool` | When `True` a `DFPMessageMeta` instance will be constructed for each user not excluded by the `skip_users` and `only_users` filters |
| `skip_users` | `List[str]` or `None` | List of users to exclude, when `include_generic` is `True` excluded records will also be excluded from the generic user. Mutually exclusive with `only_users`.  |
| `only_users` | `List[str]` or `None` | Limit records to a specific list of users, when `include_generic` is `True` the generic user's records will also be limited to the users in this list. Mutually exclusive with `skip_users`. |

#### Rolling Window Stage (`DFPRollingWindowStage`)
The `DFPRollingWindowStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_rolling_window_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_rolling_window_stage.py)) stage performs several key pieces of functionality for DFP.
1. This stage keeps a moving window of logs on a per user basis
   * These logs are saved to disk to reduce memory requirements between logs from the same user
1. It only emits logs when the window history requirements are met
   * Until all of the window history requirements are met, no messages will be sent to the rest of the pipeline.
   * See the below options for configuring the window history requirements
1. It repeats the necessary logs to properly calculate log dependent features.
   * To support all column feature types, incoming log messages can be combined with existing history and sent to downstream stages.
   * For example, to calculate a feature that increments a counter for the number of logs a particular user has generated in a single day, we must have the user's log history for the past 24 hours. To support this, this stage will combine new logs with existing history into a single `DataFrame`.
   * It is the responsibility of downstream stages to distinguish between new logs and existing history.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `min_history` | `int` | Exclude users with less than `min_history` records, setting this to `1` effectively disables this feature |
| `min_increment` | `int` | Exclude incoming batches for users where less than `min_increment` new records have been added since the last batch, setting this to `0` effectively disables this feature |
| `max_history` | `int`, `str` or `None` | When not `None`, include up to `max_history` records. When `max_history` is an int, then the last `max_history` records will be included. When `max_history` is a `str` it is assumed to represent a duration parsable by [`pandas.Timedelta`](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html) and only those records within the window of [latest timestamp - `max_history`, latest timestamp] will be included. |
| `cache_dir` | `str` | Optional path to cache directory, cached items will be stored in a subdirectory under `cache_dir` named `rolling-user-data` this directory, along with `cache_dir` will be created if it does not already exist. |

> **Note:**  this stage computes a row hash for the first and last rows of the incoming `DataFrame` as such all data contained must be hashable, any non-hashable values such as `lists` should be dropped or converted into hashable types in the `DFPFileToDataFrameStage`.

#### Preprocessing Stage (`DFPPreprocessingStage`)
The `DFPPreprocessingStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_preprocessing_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_preprocessing_stage.py)) stage, the actual logic of preprocessing is defined in the `input_schema` argument.  Since this stage occurs in the pipeline after the `DFPFileBatcherStage` and `DFPSplitUsersStage` stages all records in the incoming `DataFrame` correspond to only a single user within a specific time period allowing for columns to be computer on a per-user per-time period basis such as the `logcount` and `locincrement` features mentioned above.  Making the type of processing performed in this stage different from those performed in the `DFPFileToDataFrameStage`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `input_schema` | `DataFrameInputSchema` | Schema specifying columns to be included in the output `DataFrame` including computed columns |

## Training Pipeline
![Training PipelineOverview](img/dfp_training_overview.png)

Training must begin with the generic user model​ which is trained with the logs from all users.  This model serves as a fallback model for users & accounts without sufficient training data​.  The name of the generic user is defined in the `ae.fallback_username` attribute of the Morpheus config object and defaults to `generic_user`.

After training the generic model, individual user models can be trained​.  Individual user models provide better accuracy but require sufficient data​, many users do not have sufficient data to accurately train the model​.

### Training Stages

#### Training Stage (`DFPTraining`)
The `DFPTraining` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_training.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_training.py)) trains a model for each incoming `DataFrame` and emits an instance of `morpheus.messages.multi_ae_message.MultiAEMessage` containing the trained model.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `model_kwargs` | `dict` or `None` | Optional dictionary of keyword arguments to be used when constructing the model.  See [`AutoEncoder`](https://github.com/nv-morpheus/dfencoder/blob/master/dfencoder/autoencoder.py) for information on the available options.|

#### MLflow Model Writer Stage (`DFPMLFlowModelWriterStage`)
The `DFPMLFlowModelWriterStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_mlflow_model_writer.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_mlflow_model_writer.py)) stage publishes trained models into MLflow, skipping any model which lacked sufficient training data (current required minimum is 300 log records).

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `model_name_formatter` | `str` | Optional format string to control the name of models stored in MLflow, default is `dfp-{user_id}`. Currently available field names are: `user_id` and `user_md5` which is an md5 hexadecimal digest as returned by [`hash.hexdigest`](https://docs.python.org/3.8/library/hashlib.html?highlight=hexdigest#hashlib.hash.hexdigest). |
| `experiment_name_formatter` | `str` | Optional format string to control the experiment name for models stored in MLflow, default is `/dfp-models/{reg_model_name}`.  Currently available field names are: `user_id`, `user_md5` and `reg_model_name` which is the model name as defined by `model_name_formatter` once the field names have been applied. |
| `databricks_permissions` | `dict` or `None` | Optional, when not `None` sets permissions needed when using a databricks hosted MLflow server |

> **Note:**  If using a remote MLflow server, users will need to call [`mlflow.set_tracking_uri`](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) before starting the pipeline.

## Inference Pipeline
![Inference Pipeline Overview](img/dfp_inference_overview.png)

### Inference Stages

#### Inference Stage (`DFPInferenceStage`)
The `DFPInferenceStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_inference_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_inference_stage.py)) stage loads models from MLflow and performs inferences against those models.  This stage emits a message containing the original `DataFrame` along with new columns containing the z score (`mean_abs_z`), along with the name and version of the model that generated that score (`model_version`).  For each feature in the model three additional columns will also be added:
* `<feature name>_loss` : The loss
* `<feature name>_z_loss` : The loss z-score
* `<feature name>_pred` : The predicted value

For a hypothetical feature named `result` the three added columns will be: `result_loss`, `result_z_loss`, `result_pred`.

For performance models fetched from MLflow are cached locally, and are cached for up to 10 minutes allowing updated models to be routinely updated.  In addition to caching individual models the stage also maintains a cache of which models are available, so a newly trained user model published to MLflow won't be visible to an already running inference pipeline for up to 10 minutes.

For any user without an associated model in MLflow, the model for the generic user is used. The name of the generic user is defined in the `ae.fallback_username` attribute of the Morpheus config object defaults to `generic_user`.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `model_name_formatter` | `str` | Format string to control the name of models fetched from MLflow.  Currently available field names are: `user_id` and `user_md5` which is an md5 hexadecimal digest as returned by [`hash.hexdigest`](https://docs.python.org/3.8/library/hashlib.html?highlight=hexdigest#hashlib.hash.hexdigest). |



#### Post Processing Stage (`DFPPostprocessingStage`)
The `DFPPostprocessingStage` ([examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_postprocessing_stage.py](/examples/digital_fingerprinting/production/morpheus/dfp/stages/dfp_postprocessing_stage.py)) stage filters the output from the inference stage for any anomalous messages. Logs which exceed the specified Z-Score will be passed onto the next stage.  All remaining logs which are below the threshold will be dropped.

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `c` | `morpheus.config.Config` | Morpheus config object |
| `z_score_threshold` | `float` | Optional, sets the threshold value above which values of `mean_abs_z` must be above in order to be considered  an anomaly, default is 2.0 |
