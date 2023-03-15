## DFP Inference Pipe Module

This module function allows for the consolidation of multiple dfp pipeline modules relevant to the inference process into a single module.

### Configurable Parameters

- **batching_options** (dict): Options for batching the data, including start and end times, sampling rate, and other settings.
- **cache_dir** (string): Directory to cache the rolling window data.
- **detection_criteria** (dict): Criteria for filtering detections, such as threshold and field_name.
- **inference_options** (dict): Options for the inference module, including model settings and other configurations.
- **num_output_ports** (int): Number of output ports for the module.
- **preprocessing_options** (dict): Options for preprocessing the data, including schema and timestamp column name.
- **stream_aggregation_options** (dict): Options for aggregating the data by stream, including aggregation span and cache settings.
- **timestamp_column_name** (string): Name of the timestamp column in the input data.
- **user_splitting_options** (dict): Options for splitting the data by user, including filtering and user ID column name.
- **write_to_file_options** (dict): Options for writing the detections to a file, such as filename and overwrite settings.

### Example JSON Configuration

```json
{
  "batching_options": {...},
  "cache_dir": "/path/to/cache",
  "detection_criteria": {...},
  "inference_options": {...},
  "num_output_ports": 2,
  "preprocessing_options": {...},
  "stream_aggregation_options": {...},
  "timestamp_column_name": "timestamp",
  "user_splitting_options": {...},
  "write_to_file_options": {...}
}
```