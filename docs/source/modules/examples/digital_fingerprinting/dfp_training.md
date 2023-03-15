# DFP Training Module

This module function is responsible for training the model.

## Configurable Parameters

- `feature_columns` (list): List of feature columns to train on.
- `epochs` (int): Number of epochs to train for.
- `model_kwargs` (dict): Keyword arguments to pass to the model (see dfencoder.AutoEncoder).
- `validation_size` (float): Size of the validation set.

## JSON Example

```json
{
  "feature_columns": [
    "column1",
    "column2",
    "column3"
  ],
  "epochs": 50,
  "model_kwargs": {
    "encoder_layers": [
      64,
      32
    ],
    "decoder_layers": [
      32,
      64
    ],
    "activation": "relu",
    "swap_p": 0.1,
    "lr": 0.001,
    "lr_decay": 0.9,
    "batch_size": 32,
    "verbose": 1,
    "optimizer": "adam",
    "scalar": "min_max",
    "min_cats": 10,
    "progress_bar": false,
    "device": "cpu"
  },
  "validation_size": 0.1
}
```