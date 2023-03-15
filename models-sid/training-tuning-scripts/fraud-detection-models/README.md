<!--
SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->



## Instruction how to train new GNN models. 

### Setup training environment

Install packages for training GNN model. 

```
pip install -r requirements.txt
```

### Options for training and tuning models.

```
python training.py --help
optional arguments:
  -h, --help            show this help message and exit
  --training-data TRAINING_DATA
                     CSV with fraud_label
  --validation-data VALIDATION_DATA
                        CSV with fraud_label
  --epochs EPOCHS     Number of epochs
  --node_type NODE_TYPE
                        Target node type
  --output-xgb OUTPUT_XGB
                        output file to save xgboost model
  --output-hinsage OUTPUT_HINSAGE
                        output file to save GraphHinSage model
  --save_model SAVE_MODEL
                        Save models to given  filenames
  --embedding_size EMBEDDING_SIZE
                        output file to save new model

```


#### Example usage:

```bash
export DATASET=../../dataset

python training.py --training-data $DATASET/training-data/fraud-detection-training-data.csv \
--validation-data $DATASET\validation-datafraud-detection-validation-data.csv \
         --epoch 10 \
         --output-xgb model/xgb.pt \ 
         --output-hinsage model/hinsage.pt \
         --save_model True
```

This results in a trained models of GraphSAGE (hinsage.pt) and Gradient boosting tree (xgb.pt) at the `model` directory.
