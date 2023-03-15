## DFP Split Users Module

This module function splits the data based on user IDs.

### Configurable Parameters

- `fallback_username`: The user ID to use if the user ID is not found (string, default: 'generic_user')
- `include_generic`: Whether to include a generic user ID in the output (boolean, default: `False`)
- `include_individual`: Whether to include individual user IDs in the output (boolean, default: `False`)
- `only_users`: List of user IDs to include in the output; other user IDs will be excluded (list, default: `[]`)
- `skip_users`: List of user IDs to exclude from the output (list, default: `[]`)
- `timestamp_column_name`: Name of the column containing timestamps (string, default: 'timestamp')
- `userid_column_name`: Name of the column containing user IDs (string, default: 'username')

### Example JSON Configuration

```json
{
  "fallback_username": "generic_user",
  "include_generic": false,
  "include_individual": true,
  "only_users": ["user1", "user2", "user3"],
  "skip_users": ["user4", "user5"],
  "timestamp_column_name": "timestamp",
  "userid_column_name": "username"
}
```