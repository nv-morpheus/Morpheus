## DFP Training Pipe Module

This module function consolidates multiple DFP pipeline modules relevant to the training process into a single module.

### Configurable Parameters

- `timestamp_column_name` (str): Name of the timestamp column used in the data.
- `cache_dir` (str): Directory to cache the rolling window data.
- `batching_options` (dict): Options for batching files.
    - `end_time` (str): End time of the time range to process.
    - `iso_date_regex_pattern` (str): ISO date regex pattern.
    - `parser_kwargs` (dict): Keyword arguments to pass to the parser.
    - `period` (str): Time period to batch the data.
    - `sampling_rate_s` (float): Sampling rate in seconds.
    - `start_time` (str): Start time of the time range to process.
- `user_splitting_options` (dict): Options for splitting data by user.
    - `fallback_username` (str): Fallback user to use if no model is found for a user.
    - `include_generic` (bool): Include generic models in the results.
    - `include_individual` (bool): Include individual models in the results.
    - `only_users` (List[str]): List of users to include in the results.
    - `skip_users` (List[str]): List of users to exclude from the results.
    - `userid_column_name` (str): Column name for the user ID.
- `stream_aggregation_options` (dict): Options for aggregating data by stream.
    - `timestamp_column_name` (str): Name of the column containing timestamps.
    - `cache_mode` (str): Cache mode to use.
    - `trigger_on_min_history` (bool): Trigger on minimum history.
    - `aggregation_span` (str): Aggregation span.
    - `trigger_on_min_increment` (bool): Trigger on minimum increment.
    - `cache_to_disk` (bool): Cache to disk.
- `preprocessing_options` (dict): Options for preprocessing the data.
- `dfencoder_options` (dict): Options for configuring the data frame encoder, used for training the model.
- `mlflow_writer_options` (dict): Options for the MLflow model writer, which is responsible for saving the trained
  model.

### Example JSON Configuration

```json
{
  "timestamp_column_name": "timestamp",
  "cache_dir": "/tmp/cache",
  "batching_options": {
    "end_time": "2023-03-01T00:00:00",
    "iso_date_regex_pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}",
    "parser_kwargs": {},
    "period": "1min",
    "sampling_rate_s": 60,
    "start_time": "2023-02-01T00:00:00"
  },
  "user_splitting_options": {
    "fallback_username": "generic",
    "include_generic": true,
    "include_individual": true,
    "only_users": [],
    "skip_users": [],
    "userid_column_name": "user_id"
  },
  "stream_aggregation_options": {
    "timestamp_column_name": "timestamp",
    "cache_mode": "memory",
    "trigger_on_min_history": 1,
    "aggregation_span": "1min",
    "trigger_on_min_increment": 1,
    "cache_to_disk": false
  },
  "preprocessing_options": {
    "enable_task_filtering": true,
    "filter_task_type": "taskA",
    "enable_data_filtering": true,
    "filter_data_type": "typeA"
  },
  "dfencoder_options": {
    "feature_columns": [
      "column1",
      "column2"
    ],
    "epochs": 10,
    "validation_size": 0.2,
    "model_kwargs": {
      "encoder_layers": [
        128,
        64
      ],
      "decoder_layers": [
        64,
        128
      ],
      "activation": "relu",
      "swap_p": 0.1,
      "lr": 0.001,
      "lr_decay": 0.99,
      "batch_size": 256,
      "verbose": 1,
      "optimizer": "adam",
      "scalar": "minmax",
      "min_cats": 2,
      "progress_bar": true,
      "device": "cpu"
    }
  },
  "mlflow_writer_options": {
    "model_name_formatter": "trained_model_{timestamp}",
    "experiment_name_formatter": "training_experiment_{timestamp}",
    "conda_env": "path/to/conda_env.yml",
    "timestamp_column_name": "timestamp",
    "databricks_permissions": {
      "read_users": [
        "user1",
        "user2"
      ],
      "write_users": [
        "user1"
      ],
      "manage_users": [
        "user1"
      ]
    }
  }
}
```