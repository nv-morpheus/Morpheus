

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
