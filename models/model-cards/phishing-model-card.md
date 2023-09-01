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


# Phishing Detection

# Model Overview

## Description:
* Phishing detection is a binary classifier differentiating between phishing/spam and benign emails and SMS messages. <br>

## References(s):
* https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection <br>
* Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.04805 <br> 

## Model Architecture: 

**Architecture Type:** 

* Transformers <br>

**Network Architecture:** 

* BERT <br>

## Input: (Enter "None" As Needed)

**Input Format:** 

* Evaluation script downloads the smsspamcollection.zip and extract tabular information into a dataframe <br>

**Input Parameters:** 

* SMS/emails <br>

**Other Properties Related to Output:** 

* N/A <br>

## Output: (Enter "None" As Needed)

**Output Format:** 

* Binary Results, Fraudulent or Benign <br>

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

* http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Dataset consists of SMSs <br>

**Dataset License:** 

* https://creativecommons.org/licenses/by/4.0/legalcode taken from https://archive.ics.uci.edu/dataset/228/sms+spam+collection <br>

## Evaluation Dataset:

**Link:** 

* https://github.com/nv-morpheus/Morpheus/blob/branch-23.11/models/datasets/validation-data/phishing-email-validation-data.jsonlines  <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 

* Dataset consists of SMSs <br>

**Dataset License:** 

* https://creativecommons.org/licenses/by/4.0/legalcode taken from https://archive.ics.uci.edu/dataset/228/sms+spam+collection <br>

## Inference:

**Engine:** 

* Triton <br>

**Test Hardware:** <br>

* Other  <br>

# Subcards

## Model Card ++ Bias Subcard

### What is the gender balance of the model validation data?  

* Not Applicable

### What is the racial/ethnicity balance of the model validation data?

* Not Applicable

### What is the age balance of the model validation data?

* Not Applicable

### What is the language balance of the model validation data?

* English

### What is the geographic origin language balance of the model validation data?

* UK

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
* The model is primarily designed for testing purposes and serves as a small pre-trained model specifically used to evaluate and validate the phishing detection pipeline. Its application is focused on assessing the effectiveness of the pipeline rather than being intended for broader use cases or specific applications beyond testing.

### Fill in the blank for the model technique.

* This model is designed for developers seeking to test the phishing detection pipeline with a small pre-trained model.

### Name who is intended to benefit from this model. 

* The intended beneficiaries of this model are developers who aim to test the performance and functionality of the phishing pipeline using synthetic datasets. It may not be suitable or provide significant value for real-world phishing messages. 

### Describe the model output. 
* This model output can be used as a binary result, Phishing/Spam or Benign 

### List the steps explaining how this model works.  
* A BERT model gets fine-tuned with the dataset and in the inference it predicts one of the binary classes. Phishing/Spam or Benign.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* For different email/SMS types and content, different models need to be trained.

### What performance metrics were used to affirm the model's performance?
* F1

### What are the potential known risks to users and stakeholders?
* N/A

### What training is recommended for developers working with this model?
* Familiarity with the Morpheus SDK is recommended for developers working with this model.

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.
* http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* None

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

* The primary application for this model is testing the Morpheus phishing detection pipeline

### Name use case restrictions for the model.
* This pretrained model's use case is restricted to testing the Morpheus pipeline and may not be suitable for other applications.

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
* Unknown

### Is a mechanism in place to honor data subject right of access or deletion of personal data?

* N/A

### If PII collected for the development of this AI model, was it minimized to only what was required? 
* N/A

### Is data in dataset traceable?
* N/A

### Scanned for malware?
* No

### Are we able to identify and trace source of dataset?
* N/A

### Does data labeling (annotation, metadata) comply with privacy laws?
* N/A

### Is data compliant with data subject requests for data correction or removal, if such a request was made?
* N/A
