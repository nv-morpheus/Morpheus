## DFP Data Prep Module

This module function prepares data for either inference or model training.

### Configurable Parameters

- `schema`: (dict)
    - `schema_str`: (str) Serialized string representing the schema.
    - `encoding`: (str) Encoding type for the schema_str.
    - `input_message_type`: (str) Pickled message type.
- `timestamp_column_name`: Name of the timestamp column (string, default: 'timestamp')

### Example JSON Configuration

```json
{
  "schema": {
    "schema_str": "cPickle schema string",
    "encoding": "latin1",
    "input_message_type": "message type"
  },
  "timestamp_column_name": "timestamp"
}
```