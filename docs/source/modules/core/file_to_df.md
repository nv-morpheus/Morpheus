## File to DataFrame Module

This module reads data from the batched files into a dataframe after receiving input from the "FileBatcher" module. In addition to loading data from the disk, it has the ability to load the file content from S3 buckets.

### Configurable Parameters

- **cache_dir** (str): Directory to cache the rolling window data.
- **file_type** (str): Type of the input file.
- **filter_null** (bool): Whether to filter out null values.
- **parser_kwargs** (dict): Keyword arguments to pass to the parser.
- **schema** (dict): Schema of the input data.
- **timestamp_column_name** (str): Name of the timestamp column.

### Example JSON Configuration

```json
{
  "cache_dir": "/path/to/cache",
  "file_type": "csv",
  "filter_null": true,
  "parser_kwargs": {
    "delimiter": ","
  },
  "schema": {
    "column1": "float",
    "column2": "float"
  },
  "timestamp_column_name": "timestamp"
}
```