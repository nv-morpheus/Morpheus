# Anomalous Behavior Profiling (ABP)

# Model Overview

## Description:
* This model is an example of a binary XGBoost classifier to differentiate between anomalous GPU behavior, such as crypto mining / GPU malware, and non-anomalous GPU-based workflows (e.g., ML/DL training). The model is an XGBoost model. <br>

## References(s):
* Chen, Guestrin (2016) XGBoost. A scalable tree boosting system. https://arxiv.org/abs/1603.02754  <br> 

## Model Architecture: 
**Architecture Type:** 
* Gradient boosting <br>
**Network Architecture:** 
* XGBOOST <br>

## Input: (Enter "None" As Needed)
**Input Format:** 
* nvidia-smi output <br>
**Input Parameters:** 
* GPU statistics that are included in the nvidia-smi output <br>
**Other Properties Related to Output:** N/A <br>

## Output: (Enter "None" As Needed)
**Output Format:** 
* Binary Results <br>
**Output Parameters:** 
* N/A <br>
**Other Properties Related to Output:** 
* N/A <br> 

## Software Integration:
**Runtime(s):** 
* Morpheus  <br>

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux <br>

## Model Version(s): 
* [v1]  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/training-data/abp-sample-nvsmi-training-data.json  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* Sample dataset consists of over 1000 nvidia-smi outputs <br>
**Dataset License:** 
* N/A <br>

## Evaluation Dataset:
**Link:** 
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/validation-data/abp-validation-data.jsonlines  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* Sample dataset consists of over 1000 nvidia-smi outputs <br>
**Dataset License:** 
* N/A <br>

## Inference:
**Engine:** 
* [Triton] <br>
**Test Hardware:** <br>
* [Other]  <br>

# Subcards

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

### What is the face/key point balance of the model validation data? 

* Not Applicable

### What is the skin/tone balance of the model validation data?

* Not Applicable

### What is the religion balance of the model validation data?

* Not Applicable

### Individuals from the following adversely impacted (protected classes) groups participate in model design and testing.

* Not Applicable

### Describe measures taken to mitigate against unwanted bias.

* Not Applicable

## Model Card ++ Explainability Subcard

### Name example applications and use cases for this model. 
* This model is intended to be used in data centers with GPUs for anomalous behavior profiling use cases to detect crypto mining. It's trained with a very small sample of data, other models will be required for more differentiating between other GPU workloads and other crypto mining methods.

### Fill in the blank for the model technique.

* This model is intended for developers that want to build and/or customize XGBoost models to detect crypto mining on GPUs.

### Name who is intended to benefit from this model. 

* This model is intended for users that use gradient boosting models for detecting crypto mining

### Describe the model output. 
* This model output can be used as a binary result, Crypto mining or legitimate GPU usage. 

### List the steps explaining how this model works.  
* nvidia-smi features are used as the input and the model predicts a label for each row 

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* For different GPU workloads different models need to be trained.


### What performance metrics were used to affirm the model's performance?
* Accuracy

### What are the potential known risks to users and stakeholders?
* N/A

### What training is recommended for developers working with this model?
* None

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/training-data/abp-sample-nvsmi-training-data.json

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

* Anomalous Behavior Profiling applications for GPUs

### Name use case restrictions for the model.
* Different models need to be trained for other types GPU workloads which would generate different GPU statistics.

### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.  
* N/A

### Technical robustness and model security validated?
* No

### Is the model and dataset compliant with National Classification Management Society (NCMS)?
* No

### Are there explicit model and dataset restrictions?
* No

### Are there access restrictions to systems, model, and data?
* N/A

### Is there a digital signature?

* No

## Model Card ++ Privacy Subcard


### Generatable or reverse engineerable personally-identifiable information (PII)?
* Neither

### Was consent obtained for any PII used?
* N/A

### Protected classes used to create this model? (The following were used in model the model's training:)
* N/A

### How often is dataset reviewed?
* N/A

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* N/A

### Scanned for malware?
* N/A

### Are we able to identify and trace source of dataset?
* N/A

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A
