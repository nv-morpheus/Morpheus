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

# Model Overview

## Description:
This use case is currently implemented to detect changes in users' behavior that indicate a change from a human to a machine or a machine to a human. The model architecture consists of an Autoencoder, where the reconstruction loss of new log data is used as an anomaly score.

## References(s):
- https://github.com/AlliedToasters/dfencoder/blob/master/dfencoder/autoencoder.py
- Rasheed Peng Alhajj Rokne Jon: Fourier Transform Based Spatial Outlier Mining 2009 - https://link.springer.com/chapter/10.1007/978-3-642-04394-9_39

## Model Architecture:
The model architecture consists of an Autoencoder, where the reconstruction loss of new log data is used as an anomaly score.

**Architecture Type:**
* Autoencoder

**Network Architecture:**
* The network architecture of the model includes a 2-layer encoder with dimensions [512, 500] and a 1-layer decoder with dimensions [512]

## Input:
**Input Format:**
* AWS CloudTrail logs in json format

**Input Parameters:**
* None

**Other Properties Related to Output:**
* Not Applicable (N/A)

## Output:
**Output Format:**
* Anomaly score and the reconstruction loss for each feature in a pandas dataframe

**Output Parameters:**
* None

**Other Properties Related to Output:**
* Not Applicable

## Software Integration:
**Runtime(s):**
* Morpheus

**Supported Hardware Platform(s):** <br>
* Ampere/Turing <br>

**Supported Operating System(s):** <br>
* Linux<br>

## Model Version(s):
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/dfp-models/hammah-role-g-20211017-dill.pkl
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/dfp-models/hammah-user123-20211017-dill.pkl

# Training & Evaluation:

## Training Dataset:

**Link:**
* https://github.com/nv-morpheus/Morpheus/tree/branch-23.11/models/datasets/training-data/cloudtrail

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

The training dataset consists of AWS CloudTrail logs. It contains logs from two entities, providing information about their activities within the AWS environment.
* [hammah-role-g-training-part1.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/cloudtrail/hammah-role-g-training-part1.json): 700 records <br>
* [hammah-role-g-training-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/cloudtrail/hammah-role-g-training-part2.json): 1187 records <br>
* [hammah-user123-training-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/cloudtrail/hammah-user123-training-part2.json): 1000 records <br>
* [hammah-user123-training-part3.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/cloudtrail/hammah-user123-training-part3.json): 1000 records <br>
* [hammah-user123-training-part4.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/training-data/cloudtrail/hammah-user123-training-part4.json): 387 records <br>

**Dataset License:**
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>

## Evaluation Dataset:
**Link:**
* https://github.com/nv-morpheus/Morpheus/tree/branch-23.11/models/datasets/validation-data/cloudtrail <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):**

The evaluation dataset consists of AWS CloudTrail logs. It contains logs from two entities, providing information about their activities within the AWS environment.
* [hammah-role-g-validation.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/cloudtrail/hammah-role-g-validation.json): 314 records
* [hammah-user123-validation-part1.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part1.json): 300 records
* [hammah-user123-validation-part2.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part2.json): 300 records
* [hammah-user123-validation-part3.json](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/cloudtrail/hammah-user123-validation-part3.json): 247 records

**Dataset License:**
*  [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>

## Inference:
**Engine:**
* PyTorch

**Test Hardware:**
* Other

# Subcards

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?
* Not Applicable

### What is the racial/ethnicity balance of the model validation data?
* Not Applicable

### What is the age balance of the model validation data?
* Not Applicable

### What is the language balance of the model validation data?
* English (cloudtrail logs): 100%

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
* The model is primarily designed for testing purposes and serves as a small pretrained model specifically used to evaluate and validate the DFP pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.
* This model is designed for developers seeking to test the DFP pipeline with a small pretrained model trained on a synthetic dataset.

### Name who is intended to benefit from this model.
* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the DFP pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world cloudtrail logs analysis.

### Describe the model output.
* The model calculates an anomaly score for each input based on the reconstruction loss obtained from the trained Autoencoder. This score represents the level of anomaly detected in the input data. Higher scores indicate a higher likelihood of anomalous behavior.
* The model provides the reconstruction loss of each feature to facilitate further testing and debugging of the pipeline.

### List the steps explaining how this model works.
* The model works by training on baseline behaviors and subsequently detecting deviations from the established baseline, triggering alerts accordingly.
* [Training notebook](https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/training-tuning-scripts/dfp-models/hammah-20211017.ipynb)

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model.
* The model expects cloudtrail logs with specific features that match the training dataset. Data lacking the required features or requiring a different feature set may not be compatible with the model.

### What performance metrics were used to affirm the model's performance?
* The model's performance was evaluated based on its ability to correctly identify anomalous behavior in the synthetic dataset during testing.

### What are the potential known risks to users and stakeholders?
* None

### What training is recommended for developers working with this model?  If none, please state "none."
* Familiarity with the Morpheus SDK is recommended for developers working with this model.

### Link the relevant end user license agreement
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository (if able to share).
* https://github.com/nv-morpheus/Morpheus/tree/branch-23.11/models/datasets/training-data/cloudtrail

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.
* The primary application for this model is testing the Morpheus pipeline.

### Name use case restrictions for the model.
* The model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

### Has this been verified to have met prescribed quality standards?
* No

### Name target quality Key Performance Indicators (KPIs) for which this has been tested.
* None

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
* The synthetic data used in this model is generated using the [faker](https://github.com/joke2k/faker/blob/master/LICENSE.txt)  python package. The user agent field is generated by faker, which pulls items from its own dataset of fictitious values (located in the linked repo). Similarly, the event source field is randomly chosen from a list of event names provided in the AWS documentation. There are no privacy concerns or PII involved in this synthetic data generation process.

### Protected classes used to create this model? (The following were used in model the model's training:)
* Not applicable

### How often is dataset reviewed?
* The dataset is initially reviewed upon addition, and subsequent reviews are conducted as needed or upon request for any changes.

### Is a mechanism in place to honor data subject right of access or deletion of personal data?
* No (as the dataset is fully synthetic)

### If PII collected for the development of this AI model, was it minimized to only what was required?
* Not Applicable (no PII collected)

### Is data in dataset traceable?
* No

### Scanned for malware?
* No

### Are we able to identify and trace source of dataset?
* Yes ([fully synthetic dataset](https://github.com/nv-morpheus/Morpheus/tree/branch-23.11/models/datasets/training-data/cloudtrail))

### Does data labeling (annotation, metadata) comply with privacy laws?
* Not applicable (as the dataset is fully synthetic)

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* Not applicable (as the dataset is fully synthetic)
