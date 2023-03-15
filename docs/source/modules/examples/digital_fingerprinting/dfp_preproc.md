## DFP Preprocessing Module

This module function consolidates multiple DFP pipeline modules relevant to the inference/training process into a single module.

### Configurable Parameters

- **cache_dir** (string): Directory used for caching intermediate results.
- **timestamp_column_name** (string): Name of the column containing timestamps.
- **pre_filter_options** (dict): Options for pre-filtering control messages.
- **batching_options** (dict): Options for batching files.
- **user_splitting_options** (dict): Options for splitting data by user.
- **supported_loaders** (dict): Supported data loaders for different file types.

### Example JSON Configuration

```json
{
  "cache_dir": "/path/to/cache",
  "timestamp_column_name": "timestamp",
  "pre_filter_options": {
    "option1": "value1",
    "option2": "value2"
  },
  "batching_options": {
    "option1": "value1",
    "option2": "value2"
  },
  "user_splitting_options": {
    "option1": "value1",
    "option2": "value2"
  },
  "supported_loaders": {
    "file_type_1": "loader_1",
    "file_type_2": "loader_2"
  }
}
```