# Morpheus Datasets

Small datasets for testing training scripts, inference scripts, and pipelines.

## Anomalous Behavioral Profiling (ABP)

This is a labeled dataset of 1241 nv-smi logs generated once per minute from a single Tesla V100 in our lab environment running either GPU malware or benign workflows.

### Sample Training Data

- [abp-sample-nvsmi-training-data.json](./training-data/abp-sample-nvsmi-training-data.json)

### Pipeline Validation Data
The same data in both csv and jsonlines

- [abp-validation-data.csv](./validation-data/abp-validation-data.csv)
- [abp-validation-data.jsonlines](./validation-data/abp-validation-data.jsonlines)


## Digital Fingerprinting (DFP) Data

### DFP Azure Logs
This is a synthetic dataset of Azure AD logs with activities of 20 accounts (85 applications involved, 3567 records in total). The activities are split to a train and an inference set. An anomaly is included in the inference set for model validation. The data was generated using the python [faker](https://faker.readthedocs.io/en/master/#) package. If there is any resemblance to real individuals, it is purely coincidental.

#### Sample Training Data
- 3239 records in total
- Time range: 2022/08/01 - 2022/08/29
- Users' log distribution:
    - 5 high volume (>= 300) users
    - 15 medium volume (~100) users
    - 5 light volume (~10) users

- [training-data/azure/azure-ad-logs-sample-training-data.json](./training-data/azure/azure-ad-logs-sample-training-data.json)

#### Pipeline Validation Data
Data for the pipeline validation contains an anomlous activity for a single user.

- Account: `attacktarget@domain.com`    
- Time: 2022/08/31
- Description: 
    - Anomalously high log volume (100+)
    - New IP for the account
    - New location for the account (new country, state, city, latitude, longitude)
    - New browser
    - New app access (80 new apps accessed by the account on the day)

This dataset is stored in our S3 bucket. It can be downloaded using a script.
- [fetch_example_data.py](../../examples/digital_fingerprinting/fetch_example_data.py)


### DFP Cloudtrail Logs

This is a synthetic dataset of AWS CloudTrail logs events with activities from 2 entities/users in separate files. 

Files for `user-123` include a single csv and split json versions of the same data:
#### Sample Training Data
- [dfp-cloudtrail-user123-training-data.csv](./training-data/dfp-cloudtrail-user123-training-data.csv)
- [hammah-user123-training-part2.json](./training-data/cloudtrail/hammah-user123-training-part2.json)
- [hammah-user123-training-part3.json](./training-data/cloudtrail/hammah-user123-training-part3.json)
- [training-data/cloudtrail/hammah-user123-training-part4.json](./training-data/cloudtrail/hammah-user123-training-part4.json)

#### Pipeline Validation Data
- [dfp-cloudtrail-user123-validation-data-input.csv](./validation-data/dfp-cloudtrail-user123-validation-data-input.csv)
- [dfp-cloudtrail-user123-validation-data-output.csv](./validation-data/dfp-cloudtrail-user123-validation-data-input.csv)

Files for `role-g` include a single csv and split json version of the same data:
#### Sample Training Data
- [dfp-cloudtrail-role-g-training-data.csv](./training-data/dfp-cloudtrail-role-g-training-data.csv)
- [hammah-role-g-training-part1.json](./training-data/cloudtrail/hammah-role-g-training-part1.json)
- [hammah-role-g-training-part2.json](./training-data/cloudtrail/hammah-role-g-training-part1.json)

#### Pipeline Validation Data
- [dfp-cloudtrail-role-g-validation-data-input.csv](.validation-data/dfp-cloudtrail-role-g-validation-data-input.csv)
- [dfp-cloudtrail-role-g-validation-data-output.csv](.validation-data/dfp-cloudtrail-role-g-validation-data-output.csv)


## Fraud Detection

This is a small dataset augmented from the artificially generated transaction network demo data from the authors of [Inductive Graph Representation Learning for Fraud Detection](https://www.researchgate.net/publication/357706343_Inductive_Graph_Representation_Learning_for_fraud_detection). The original demo data of 753 labeled transactions was downloaded from the paper's [github repo](https://github.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection/blob/master/Demo/demo_ccf.csv) on 02/10/2022 with an MD5 hash `64af64fcc6e3d55d25111a3f257378a4`. We augmented the training dataset to increase benign transactions by replicating that portion of the dataset for a total of 12053 transactions.

### Sample Training Data
- [fraud-detection-training-data.csv](./training-data/fraud-detection-training-data.csv)

### Pipeline Validation Data
- [fraud-detection-validation-data.csv](./validation-data/fraud-detection-validation-data.csv)


## Log Parsing

This sample dataset consists of a subset of Apache logs collected from a linux system running Apache Web server as part of a larger public log dataset on [loghub](https://github.com/logpai/loghub/blob/master/Apache/Apache_2k.log). The file was downloaded on 01/14/2020 with an MD5 hash of `1c3a706386b3ebc03a2ae07a2d864d66`. The logs were parsed using an apache log parsing [package](https://github.com/amandasaurus/apache-log-parser) to create a labeled dataset.


### Sample Training Data

- [log-parsing-training-data.csv](./training-data/log-parsing-training-data.csv)

### Pipeline Validation Data
Log validation data in csv and json format

- [log-parsing-validation-data-input.csv](./validation-data/log-parsing-validation-data-input.csv)
- [log-parsing-validation-data-output.jsonlines](./validation-data/log-parsing-validation-data-output.jsonlines)


## Phishing Detection

The SMS Spam Collection is a public set of 5574 SMS labeled messages that have been collected for mobile phone spam research hosted at UCI Machine Learning Repository: [SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) last accessed on 11/09/2022 with an MD5 hash of `ab53f9571d479ee677e7b283a06a661a` 
During training, 20% of the dataset is randomly selected as the test set and is saved as a jsonlines file for use in pipeline validation. 

### Pipeline Validation Data
- [phishing-email-validation-data.jsonlines](./validation-data/phishing-email-validation-data.jsonlines)

### Example Data for Developer Guide
Additionally a subset of 100 messages from the dataset were augmented to include sender and recipient information using the python [faker](https://faker.readthedocs.io/en/master/#) package. If there is any resemblance to real individuals, it is purely coincidental.
- [emails_with_addresses.jsonlines](../../examples/data/email_with_addresses.jsonlines)


## Ransomware Detection via AppShield

The dataset was generated by running ransomware and benign processes in a lab environment and recording the output from several plugins from the [Volatility framework](https://github.com/volatilityfoundation/volatility3) including `cmdline`, `envars`, `handles`, `ldrmodules`, `netscan`, `pslist`, `thredlist`, `vadinfo`. The training csv file contains 530 columns- a combination of features from the Volatility Plugins. This data collection is part of [DOCA AppShield](https://developer.nvidia.com/networking/doca).

### Sample Training Data
Training data csv consists of 87968 preprocessed and labeled AppShield processes from 32 snapshots collected from 256 unique benign and ransomware activities.
- [ransomware-training-data.csv](./training-data/ransomware-training-data.csv)

### Pipeline Validation Data
The validation set contains raw data from 27 AppShield snapshots.
- [appshield data directory](../../examples/data/appshield/Heur)

## Root Cause

This dataset contains a small sample of sanitized linux kernel logs of a DGX machine prior to a hardware failure. The training dataset contains 1359 logs labeled as indicators of the root cause or not. A model trained on this set can be robust enough to correctly identify previously unseen errors from the `unseen-errors` file as a root cause as well. 

### Sample Training Data
- [root-cause-training-data.csv](./training-data/root-cause-training-data.csv)
- [root-cause-unseen-errors.csv](./training-data/root-cause-unseen-errors.csv)

### Pipeline Validation Data
- [root-cause-validation-data-input.jsonlines](./validation-data/root-cause-validation-data-input.jsonlines)


## Sensitive Information Detection (SID)

This data contains 2000 synthetic pcap payloads generated to mimic sensitive and benign data found in nested jsons from web APIs and environmental variables. Each row is labeled for the presence or absence of 10 different kinds of sensitive information. The data was generated using the python [faker](https://faker.readthedocs.io/en/master/#) package and lists of most [common passwords](https://github.com/danielmiessler/SecLists/tree/master/Passwords/Common-Credentials). If there is any resemblance to real individuals, it is purely coincidental.

### Sample Training Data
- [sid-sample-training-data.csv](./training-data/sid-sample-training-data.csv)

### Pipeline Validation Data
- [sid-validation-data.csv](./validation-data/sid-validation-data.csv)


## Disclaimer
Morpheus contributors will make every effort to keep datasets up-to-date and accurate. However, the data submitted to this repository is provided on an “as-is” basis and there is no warranty or guarantee of any kind that the information is accurate, complete, current or suitable for any particular purpose. It is the responsibility of all persons who use Morpheus to independently confirm the accuracy of the data, information, and results obtained via the Morpheus example workflows.