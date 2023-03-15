## DFP Training Module

This module function is used for model training.

### Configurable Parameters

- **feature_columns** (list): List of feature columns to train on.
- **epochs** (int): Number of epochs to train for.
- **model_kwargs** (dict): Keyword arguments to pass to the model (see `dfencoder.AutoEncoder`).
- **validation_size** (float): Size of the validation set.

### Example JSON Configuration

```json
{
  "feature_columns": ["feature_1", "feature_2", "feature_3"],
  "epochs": 100,
  "model_kwargs": {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.5,
    "activation": "relu"
  },
  "validation_size": 0.2
}
```