# Root Cause Analysis

# Model Overview

## Description:
* Root cause analysis is a binary classifier differentiating between ordinary logs and errors/problems/root causes in the log files. <br>

## References(s):
* Devlin J. et al. (2018), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.04805 <br> 

## Model Architecture: 
**Architecture Type:** 
* Transformers <br>
**Network Architecture:** 
* BERT <br>

## Input: (Enter "None" As Needed)
**Input Format:** 
* CSV <br>
**Input Parameters:** 
* kern.log file contents <br>
**Other Properties Related to Output:** 
* N/A <br>

## Output: (Enter "None" As Needed)
**Output Format:** 
* Binary Results, Root Cause or Ordinary <br>
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
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/training-data/root-cause-training-data.csv <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* kern.log files from DGX machines <br>
**Dataset License:** N/A <br>

## Evaluation Dataset:
**Link:** 
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/validation-data/root-cause-validation-data-input.jsonlines  <br>
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* kern.log files from DGX machines <br>
**Dataset License:** 
* N/A <br>

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
* This model is an example of customized transformer-based root cause analysis. It can be further fine-tuned for specific root cause analysis or predictive maintenance needs and of your enterprise using the fine-tuning scripts in the repo. The hyper parameters can be optimised to adjust to get the best results with your dataset. The aim is to get the model to predict some false positives that could be previously unknown error types. Users can use this root cause analysis method with other log types too. If they have known failures in their logs, they can use them to train along with ordinary logs and can detect other root causes they weren't aware of before.

### Fill in the blank for the model technique.

* This model is intended for developers that want to build and/or customize root cause analysis models.

### Name who is intended to benefit from this model. 

* This model is intended for users who use language models for root cause analysis.

### Describe the model output. 
* This model output can be used as a binary result, Root cause or Ordinary 

### List the steps explaining how this model works.  
* A BERT model gets fine-tuned with the kern.log dataset and in the inference it predicts one of the binary classes. Root cause or Ordinary.

### Name the adversely impacted groups (protected classes) this has been tested to deliver comparable outcomes regardless of:
* Not Applicable

### List the technical limitations of the model. 
* For different log types and content, different models need to be trained.

### What performance metrics were used to affirm the model's performance?
* F1

### What are the potential known risks to users and stakeholders?
* N/A

### What training is recommended for developers working with this model?
* None

### Link the relevant end user license agreement 
* [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)<br>


## Model Card ++ Saftey & Security Subcard

### Link the location of the training dataset's repository.
* https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/models/datasets/training-data/root-cause-training-data.csv

### Is the model used in an application with physical safety impact?
* No

### Describe physical safety impact (if present).
* N/A

### Was model and dataset assessed for vulnerability for potential form of attack?
* No

### Name applications for the model.

* Root Cause Analysis, enabling predictive maintenance. 

### Name use case restrictions for the model.
* Different models need to be trained for other types of logs

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