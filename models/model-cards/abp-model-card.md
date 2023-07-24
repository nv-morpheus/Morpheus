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

* v1  <br>

# Training & Evaluation: 

## Training Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/abp-sample-nvsmi-training-data.json  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Sample dataset consists of over 1000 nvidia-smi outputs <br>

**Dataset License:** 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) <br>

## Evaluation Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/abp-validation-data.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Sample dataset consists of over 1000 nvidia-smi outputs <br>

**Dataset License:** 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* Other <br>

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

* The model is primarily designed for testing purposes and serves as a small model specifically used to evaluate and validate the ABP pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.

* The model is primarily designed for testing purposes. This model is intended to be an example for developers that want to test Morpheus ABP pipeline.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the functionality of the ABP models for detecting crypto mining.

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

* Familiarity with the Morpheus SDK is recommended for developers working with this model.

### Link the relevant end user license agreement 

* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.

* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/abp-sample-nvsmi-training-data.json

### Is the model used in an application with physical safety impact?

* No

### Describe physical safety impact (if present).

* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?

* No

### Name applications for the model.

* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.

* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

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

* No

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

* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 

* N/A

### Is data in dataset traceable?

* N/A

### Scanned for malware?

* No

### Are we able to identify and trace source of dataset?

* Yes

### Does data labeling (annotation, metadata) comply with privacy laws?

* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?

* N/A
