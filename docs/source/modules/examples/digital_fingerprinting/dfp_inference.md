## DFP Inference Module

This module function performs the inference process.

### Configurable Parameters

- **model_name_formatter**: Formatter for model names (string).
- **fallback_username**: Fallback user to use if no model is found for a user (string).
- **timestamp_column_name**: Name of the timestamp column (string).

### Example JSON Configuration

```json
{
  "model_name_formatter": "user_{username}_model",
  "fallback_username": "generic_user",
  "timestamp_column_name": "timestamp"
}
```