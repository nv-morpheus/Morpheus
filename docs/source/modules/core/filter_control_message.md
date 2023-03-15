## Filter Control Message Module

When the requirements are met, this module gently discards the control messages.

### Configurable Parameters

- `enable_task_filtering` (bool): Enables filtering based on task type.
- `enable_data_type_filtering` (bool): Enables filtering based on data type.
- `filter_task_type` (str): The task type to be used as a filter.
- `filter_data_type` (str): The data type to be used as a filter.

### Example JSON Configuration

```json
{
  "enable_task_filtering": true,
  "enable_data_type_filtering": true,
  "filter_task_type": "specific_task",
  "filter_data_type": "desired_data_type"
}
```