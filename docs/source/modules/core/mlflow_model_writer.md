## MLflow Model Writer Module

This module uploads trained models to the MLflow server.

### Configurable Parameters

- **model_name_formatter** (str): Formatter for the model name.
- **experiment_name_formatter** (str): Formatter for the experiment name.
- **conda_env** (str): Conda environment for the model.
- **timestamp_column_name** (str): Name of the timestamp column.
- **databricks_permissions** (dict): Permissions for the model.

### Example JSON Configuration

```json
{
  "model_name_formatter": "model_name_{timestamp}",
  "experiment_name_formatter": "experiment_name_{timestamp}",
  "conda_env": "path/to/conda_env.yml",
  "timestamp_column_name": "timestamp",
  "databricks_permissions": {
    "read": ["read_user1", "read_user2"],
    "write": ["write_user1", "write_user2"]
  }
}
```