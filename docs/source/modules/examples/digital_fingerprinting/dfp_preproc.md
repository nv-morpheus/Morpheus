## dfp_preproc

This module function allows for the consolidation of multiple dfp pipeline modules relevant to inference/training
process into a single module.

### Configurable Parameters

- `cache_dir` (str): Directory used for caching intermediate results.
- `timestamp_column_name` (str): Name of the column containing timestamps.
- `pre_filter_options` (dict): Options for pre-filtering control messages.
    - `enable_task_filtering` (bool): Enables filtering based on task type.
    - `filter_task_type` (str): The task type to be used as a filter.
    - `enable_data_filtering` (bool): Enables filtering based on data type.
    - `filter_data_type` (str): The data type to be used as a filter.
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
- `supported_loaders` (dict): Supported data loaders for different file types.

### Example JSON Configuration

```json
{
  "cache_dir": "/tmp/cache",
  "timestamp_column_name": "timestamp",
  "pre_filter_options": {
    "enable_task_filtering": true,
    "filter_task_type": "task_a",
    "enable_data_filtering": true,
    "filter_data_type": "type_a"
  },
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
  "supported_loaders": {}
}
```