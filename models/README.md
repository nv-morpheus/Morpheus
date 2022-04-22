# Morpheus Models

Pretrained models for Morpheus with corresponding training/validation scripts and datasets.

## Repo Structure
Every Morpheus use case has a subfolder, **`<use-case>-models`**, that contains the model files for the use case. Training and validation datasets and scripts are also provided in [datasets](./datasets/), [training-tuning-scripts](./training-tuning-scripts/), and [validation-inference-scripts](./validation-inference-scripts/). Jupyter notebook (`.ipynb`) version of the training and fine-tuning scripts are also provided.

The `triton_model_repo` contains the necessary directory structure and configuration files in order to run the Morpheus Models in Triton Inference Server. This includes symlinks to the above-mentioned model files along with corresponding Triton config files (`.pbtxt`). More information on how to deploy this repository to Triton can be found in the [README](./triton-model-repo/README.md).

Models can also be published to an [MLflow](https://mlflow.org/) server and deployed to Triton using the [MLflow Triton plugin](https://github.com/triton-inference-server/server/tree/main/deploy/mlflow-triton-plugin). The [mlflow](./mlflow/README.md) directory contains information on how to set up a Docker container to run an MLflow server for publishing Morpheus models and deploying them to Triton.

In the root directory, the file `model-information.csv` contains the following information for each model:

 - **Model name** - Name of the model
 - **Use case** - Specific Morpheus use case the model targets
 - **Owner** - Name of the individual who owns the model
 - **Version** - Version of the model (major.minor.patch)
 - **Training epochs** - Number of epochs used during training
 - **Batch size** - Batch size used during training
 - **GPU model** - Family of GPU used during training
 - **Model accuracy** - Accuracy of the model when tested
 - **Model F1** - F1 score of the model when tested
 - **Small test set accuracy** - Accuracy of model on validation data in datasets directory
 - **Memory footprint** - Memory required by the model
 - **Input data** - Typical data that is used as input to the model
 - **Thresholds** - Values of thresholds used for validation
 - **NLP hash file** - Hash file for tokenizer vocabulary
 - **NLP max length** - Max_length value for tokenizer
 - **NLP stride** - stride value for tokenizer
 - **NLP do lower** - do_lower value for tokenizer
 - **NLP do truncate** - do_truncate value for tokenizer
 - **Version CUDA** - CUDA version used during training
 - **Version Python** - Python version used during training
 - **Version Ubuntu** - Ubuntu version used during training
 - **Version Transformers** - Transformers version used during training

## Current Use Cases Supported by Models Here
### Sensitive Information Detection
Sensitive information detection is used to identify pieces of sensitive data (e.g., AWS credentials, GitHub credentials, passwords) in unencrypted data. The model for this use case is an NLP model, specifically a transformer-based model with attention (e.g., mini-BERT).

### Anomalous Behavior Profiling
This use case is currently implemented to differentiate between crypto mining / GPU malware and other GPU-based workflows (e.g., ML/DL training). The model is a XGBoost model.

### Phishing Email Detection
This use case is currently implemented to differentiate between phishing and non-phishing emails. The models for this use case are NLP models, specifically transformer-based models with attention (e.g., BERT).

### Humans-As-Machines-Machines-As-Humans Detection
This use case is currently implemented to detect changes in users' behavior that indicate a change from a human to a machines or a machine to a human. The model is an ensemble of an autoencoder and fast fourier transform reconstruction.

### Fraud detection system Detection
This use case implemented to identify fraudulent transactions from legal transaction in credit card transaction network. The model is based on a combination of graph neural network and gradient boosting tree. It uses a bipartite heterogenous graph representation as input for GraphSAGE for feature learning and XGBoost as a classifier.