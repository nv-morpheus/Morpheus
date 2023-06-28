# Fraud Detection using Graph Neural Network

## Model Overview

### Description:
This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of `GraphSAGE` along `XGBoost` is used to identify frauds in the transaction networks. <br>

## References(s):
1. https://stellargraph.readthedocs.io/en/stable/hinsage.html?highlight=hinsage
2. https://github.com/rapidsai/clx/blob/branch-22.12/examples/forest_inference/xgboost_training.ipynb
3. Rafaël Van Belle, Charles Van Damme, Hendrik Tytgat, Jochen De Weerdt,Inductive Graph Representation Learning for fraud detection (https://www.sciencedirect.com/science/article/abs/pii/S0957417421017449)<br> 

## Model Architecture:
It uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.<br>
**Architecture Type:** Graph Neural Network and Binary classification <br>
**Network Architecture:** GraphSAGE and XGBoost <br>

## Input
Transaction data with nodes including transaction, client, and merchant.<br>
**Input Parameters:** None <br>
## Output
An anomalous score of transactions indicates a probability score of being a fraud.<br>
**Output Parameters:** None <br>
## Software Integration:
**Runtime(s):** Morpheus  <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>
  
## Model Version(s): 
* 1.0 <br>

### How To Use This Model
This model is an example of a fraud detection pipeline using a graph neural network and gradient boosting trees. This can be further retrained or fine-tuned to be used for similar types of transaction networks with similar graph structures.

# Training & Evaluation: 

## Training Dataset:

**Link:** [fraud-detection-training-data.csv](models/dataset/fraud-detection-training-data.csv)  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** A training data consists of raw 753 labeled credit card transaction data with data augmentation in a total of 12053 labeled transaction data. <br>
**Dataset License:**  (N/A) <br>

## Evaluation Dataset:
**Link:**  [fraud-detection-validation-data.csv](models/dataset/fraud-detection-validation-data.csv)  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Data consists of raw 265 labeled credit card transaction synthetically created<br>
**Dataset License:** (N/A)<br>

## Inference:
**Engine:** Triton
**Test Hardware:** <br>
* Other (Not Listed)  <br>

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?  
* Not Applicable

### What is the racial/ethnicity balance of the model validation data?
* Not Applicable

### What is the age balance of the model validation data?
* Not Applicable

### What is the language balance of the model validation data?
* Not Applicable

### What is the geographic origin language balance of the model validation data?
* Not Applicable

### What is the educational background balance of the model validation data?
* Not Applicable

### What is the accent balance of the model validation data?
* Not Applicable
### Describe measures taken to mitigate against unwanted bias.
* Not Applicable
## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* This model is intended to be used for fraud detection use cases.
### Fill in the blank for the model technique.
* This model is intended for developers that want to build and/or customize fraud detection applications using graph neural network.

### Name who is intended to benefit from this model. 
* This model is intended for users that use transactional data to build fraud detection application.

### Describe the model output.
* This model outputs fraud probability score b/n (0 & 1). 

### List the steps explaining how this model works. (e.g., )  
* The model uses a bipartite heterogeneous graph representation as input for `GraphSAGE` for feature learning and `XGBoost` as a classifier. Since the input graph is heterogeneous, a heterogeneous implementation of `GraphSAGE` (HinSAGE) is used for feature embedding.<br>

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* This model version requires a transactional data schema with entities (user, merchant, transaction) as requirement for the model.

### What performance metrics were used to affirm the model's performance?
* Area under ROC curve and Accuracy

### What are the potential known risks to users and stakeholders? 
* Not Applicable

### What training is recommended for developers working with this model?  If none, please state "none."
* none
### Link the relevant end user license agreement 
* [Apache 2.0](https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/LICENSE)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* [training dataset](models/datasets/training-data/fraud-detection-training-data.csv)

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* Not Applicable

### Was model and dataset assessed for vulnerability for potential form of attack?
* No
### Name applications for the model.
* Fraud detection in credit card transaction with the defined schema.
### Name use case restrictions for the model.
* Only tested for fraud detection in transactional data application
### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* Not Applicable
### Technical robustness and model security validated?
* Not Applicable
### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* Not Applicable

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* No
### Is there a digital signature?
* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?

* Neither

### Was consent obtained for any PII used?
* Not Applicable

### Protected classes used to create this model? (The following were used in model the model's training:)

* None of the Above


### How often is dataset reviewed?
* Other: Not Applicable

### Is a mechanism in place to honor data
* Yes
### If PII collected for the development of this AI model, was it minimized to only what was required? 
* Not applicable

### Is data in dataset traceable?
* No

### Scanned for malware?
* Not applicable
### Are we able to identify and trace source of dataset?
* Not applicable

### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable
### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable