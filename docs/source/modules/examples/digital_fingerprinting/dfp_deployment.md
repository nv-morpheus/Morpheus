<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Pipeline Module

This module function sets up a pipeline builder instance.

### Configurable Parameters

- `training_options` (dict): Options for the training pipeline module, including:
    - `timestamp_column_name` (str): Name of the timestamp column used in the data
    - `cache_dir` (str): Directory to cache the rolling window data
    - `batching_options` (dict): Options for batching the data, including:
        - `end_time` (datetime|str): End time of the time window
        - `iso_date_regex_pattern` (str): Regex pattern for ISO date matching
        - `parser_kwargs` (dict): Additional arguments for the parser
        - `period` (str): Time period for grouping files
        - `sampling_rate_s` (int): Sampling rate in seconds
        - `start_time` (datetime|str): Start time of the time window
    - `user_splitting_options` (dict): Options for splitting the data by user, including:
        - `fallback_username` (str): User ID to use if user ID not found (default: 'generic_user')
        - `include_generic` (bool): Include generic user ID in output (default: False)
        - `include_individual` (bool): Include individual user IDs in output (default: False)
        - `only_users` (list): List of user IDs to include in output, others will be excluded (default: [])
        - `skip_users` (list): List of user IDs to exclude from output (default: [])
        - `timestamp_column_name` (str): Name of column containing timestamps (default: 'timestamp')
        - `userid_column_name` (str): Name of column containing user IDs (default: 'username')
    - `stream_aggregation_options` (dict): Options for aggregating the data by stream
    - `preprocessing_options` (dict): Options for preprocessing the data
    - `dfencoder_options` (dict): Options for configuring the data frame encoder, used for training the model
    - `mlflow_writer_options` (dict): Options for the MLflow model writer, responsible for saving the trained model,
      including:
        - `model_name_formatter` (str): Format string for the model name, e.g. "model_{timestamp}"
        - `experiment_name_formatter` (str): Format string for the experiment name, e.g. "experiment_{timestamp}"
        - `timestamp_column_name` (str): Name of the timestamp column used in the data
        - `conda_env` (dict): Conda environment settings, including:
            - `channels` (list): List of channels to use for the environment
            - `dependencies` (list): List of dependencies for the environment
            - `pip` (list): List of pip packages to install in the environment
            - `name` (str): Name of the conda environment
- `inference_options` (dict): Options for the inference pipeline module, including:
    - `model_name_formatter` (str): Format string for the model name, e.g. "model_{timestamp}"
    - `fallback_username` (str): User ID to use if user ID not found (default: 'generic_user')
    - `timestamp_column_name` (str): Name of the timestamp column in the input data
    - `batching_options` (dict): Options for batching the data, including:
      [omitted for brevity]
    - `cache_dir` (str): Directory to cache the rolling window data
    - `detection_criteria` (dict): Criteria for filtering detections, such as threshold and field_name
    - `inference_options` (dict): Options for the inference module, including model settings and other configurations
    - `num_output_ports` (int): Number of output ports for the module
    - `preprocessing_options` (dict): Options for preprocessing the data, including schema and timestamp column name
    - `stream_aggregation_options` (dict): Options for aggregating the data by stream, including:
        - `aggregation_span` (int): The time span for the aggregation window, in seconds
        - `cache_to_disk` (bool): Whether to cache the aggregated data to disk
    - `user_splitting_options` (dict): Options for splitting the data by user, including:
      [omitted for brevity]
    - `write_to_file_options` (dict): Options for writing the detections to a file, such as filename and overwrite
      settings

### Example JSON Configuration

```json
{
  "training_options": {
    "timestamp_column_name": "my_timestamp",
    "cache_dir": "/path/to/cache/dir",
    "batching_options": {
      "end_time": "2023-03-17 12:00:00",
      "iso_date_regex_pattern": "YYYY-MM-DD",
      "parser_kwargs": {
        "delimiter": ","
      },
      "period": "1h",
      "sampling_rate_s": 5,
      "start_time": "2023-03-17 11:00:00"
    },
    "user_splitting_options": {
      "fallback_username": "generic_user",
      "include_generic": false,
      "include_individual": false,
      "only_users": [
        "user1",
        "user2"
      ],
      "skip_users": [
        "user3",
        "user4"
      ],
      "timestamp_column_name": "timestamp",
      "userid_column_name": "username"
    },
    "stream_aggregation_options": {
      "aggregation_span": 60,
      "cache_to_disk": true
    },
    "preprocessing_options": {
      "option1": "value1",
      "option2": "value2"
    },
    "dfencoder_options": {
      "option1": "value1",
      "option2": "value2"
    },
    "mlflow_writer_options": {
      "model_name_formatter": "model_{timestamp}",
      "experiment_name_formatter": "experiment_{timestamp}",
      "timestamp_column_name": "my_timestamp",
      "conda_env": {
        "channels": [
          "conda-forge",
          "defaults"
        ],
        "dependencies": [
          "numpy",
          "pandas"
        ],
        "pip": [
          "tensorflow==2.5.0"
        ],
        "name": "my_conda_env"
      }
    }
  },
  "inference_options": {
    "model_name_formatter": "model_{timestamp}",
    "fallback_username": "generic_user",
    "timestamp_column_name": "timestamp",
    "batching_options": {
      "end_time": "2023-03-17 14:00:00",
      "iso_date_regex_pattern": "YYYY-MM-DD",
      "parser_kwargs": {
        "delimiter": ","
      },
      "period": "1h",
      "sampling_rate_s": 5,
      "start_time": "2023-03-17 13:00:00"
    },
    "cache_dir": "/path/to/cache/dir",
    "detection_criteria": {
      "threshold": 0.5,
      "field_name": "score"
    },
    "inference_options": {
      "option1": "value1",
      "option2": "value2"
    },
    "num_output_ports": 3,
    "preprocessing_options": {
      "option1": "value1",
      "option2": "value2"
    },
    "stream_aggregation_options": {
      "aggregation_span": 60,
      "cache_to_disk": true
    },
    "user_splitting_options": {
      "fallback_username": "generic_user",
      "include_generic": false,
      "include_individual": false,
      "only_users": [
        "user1",
        "user2"
      ],
      "skip_users": [
        "user3",
        "user4"
      ],
      "timestamp_column_name": "timestamp",
      "userid_column_name": "username"
    },
    "write_to_file_options": {
      "filename": "output.txt",
      "overwrite": true
    }
  }
}
```