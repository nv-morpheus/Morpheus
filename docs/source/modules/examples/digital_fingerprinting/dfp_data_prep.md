## DFP Data Prep Module

This module function prepares data for either inference or model training.

### Configurable Parameters

- **schema**: Schema of the data (dictionary)
- **timestamp_column_name**: Name of the timestamp column (string, default: 'timestamp')

### Example JSON Configuration

```json
{
  "schema": {
    "column1": "int",
    "column2": "float",
    "column3": "string",
    "timestamp": "datetime"
  },
  "timestamp_column_name": "timestamp"
}
```