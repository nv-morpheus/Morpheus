## Serialize Module

This module filters columns from a `MultiMessage` object, emitting a `MessageMeta`.

### Configurable Parameters

- **include** (str): Regex to include columns.
- **exclude** (List[str]): List of regex patterns to exclude columns.
- **fixed_columns** (bool): If true, the columns are fixed and not determined at runtime.
- **columns** (List[str]): List of columns to include.
- **use_cpp** (bool): If true, use C++ to serialize.

### Example JSON Configuration

```json
{
  "include": "^column",
  "exclude": ["column_to_exclude"],
  "fixed_columns": true,
  "columns": ["column1", "column2", "column3"],
  "use_cpp": true
}
```