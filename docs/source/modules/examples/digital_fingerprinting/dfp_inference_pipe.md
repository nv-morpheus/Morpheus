## dfp_inference_pipe

This module function allows for the consolidation of multiple dfp pipeline modules relevant to the inference process
into a single module.

### Configurable Parameters

- `timestamp_column_name` (str): Name of the column containing timestamps.
- `cache_dir` (str): Directory used for caching intermediate results.
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
- `preprocessing_options` (dict): Options for preprocessing data.
- `inference_options` (dict): Options for configuring the inference process.
    - `model_name_formatter` (str): Formatter for the model name.
    - `fallback_username` (str): Fallback user to use if no model is found for a user.
    - `timestamp_column_name` (str): Name of the column containing timestamps.
- `detection_criteria` (dict): Criteria for filtering detections.
- `write_to_file_options` (dict): Options for writing results to a file.

### Example JSON Configuration

```json
{
  "timestamp_column_name": "timestamp",
  "cache_dir": "/tmp/cache",
  "batching_options": {
    "end_time": "2022-01-01T00:00:00Z",
    "iso_date_regex_pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z",
    "parser_kwargs": {},
    "period": "1D",
    "sampling_rate_s": 1.0,
    "start_time": "2021-01-01T00:00:00Z"
  },
  "user_splitting_options": {
    "fallback_username": "generic",
    "include_generic": true,
    "include_individual": true,
    "only_users": [
      "user_a",
      "user_b"
    ],
    "skip_users": [
      "user_c"
    ],
    "userid_column_name": "user_id"
  },
  "stream_aggregation_options": {
    "timestamp_column_name": "timestamp",
    "cache_mode": "MEMORY",
    "trigger_on_min_history": true,
    "aggregation_span": "1D",
    "trigger_on_min_increment": true,
    "cache_to_disk": false
  },
  "preprocessing_options": {},
  "inference_options": {
    "model_name_formatter": "{model_name}",
    "fallback_username": "generic",
    "timestamp_column_name": "timestamp"
  },
  "detection_criteria": {},
  "write_to_file_options": {}
}
```