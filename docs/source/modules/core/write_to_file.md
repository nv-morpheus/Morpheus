## WriteToFile Module

This module writes messages to a file.

### Configurable Parameters

- **filename** (str): Path to the output file.
- **overwrite** (bool): If true, overwrite the file if it exists.
- **flush** (bool): If true, flush the file after each write.
- **file_type** (FileTypes): Type of file to write.
- **include_index_col** (bool): If true, include the index column.

### Example JSON Configuration

```json
{
  "filename": "output.csv",
  "overwrite": true,
  "flush": false,
  "file_type": "CSV",
  "include_index_col": false
}
```