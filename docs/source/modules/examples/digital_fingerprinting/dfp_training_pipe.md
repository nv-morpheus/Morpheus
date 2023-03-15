## DFP Training Pipe Module

This module function consolidates multiple DFP pipeline modules relevant to the training process into a single module.

### Configurable Parameters

- **timestamp_column_name** (str): Name of the timestamp column used in the data.
- **cache_dir** (str): Directory to cache the rolling window data.
- **batching_options** (dict): Options for batching the data.
- **user_splitting_options** (dict): Options for splitting the data by user.
- **stream_aggregation_options** (dict): Options for aggregating the data by stream.
- **preprocessing_options** (dict): Options for preprocessing the data.
- **dfencoder_options** (dict): Options for configuring the data frame encoder, used for training the model.
- **mlflow_writer_options** (dict): Options for the MLflow model writer, which is responsible for saving the trained
  model.

### Example JSON Configuration

```json
{
  "timestamp_column_name": "timestamp",
  "cache_dir": "/path/to/cache",
  "batching_options": {
    "start_time": "2022-01-01",
    "end_time": "2022-12-31",
    "sampling_rate_s": 60,
    "period": "1d"
  },
  "user_splitting_options": {
    "userid_column_name": "username",
    "fallback_username": "generic_user",
    "include_generic": false,
    "include_individual": false,
    "only_users": [
      "user1",
      "user2"
    ],
    "skip_users": []
  },
  "stream_aggregation_options": {
    "aggregation_span": "60d",
    "cache_mode": "batch"
  },
  "preprocessing_options": {
    "schema": {
      "column1": "float",
      "column2": "float"
    }
  },
  "dfencoder_options": {
    "hidden_layers": [
      128,
      64,
      32
    ],
    "dropout_rate": 0.5,
    "activation": "relu"
  },
  "mlflow_writer_options": {
    "mlflow_tracking_uri": "http://localhost:5000",
    "experiment_name": "my_experiment",
    "artifact_root": "/path/to/artifact/root"
  }
}
```