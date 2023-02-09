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

# Morpheus Models

Morpheus comes with a number of pretrained models with corresponding training, validation scripts, and datasets. The latest release of these models can be found [here](https://github.com/nv-morpheus/Morpheus/blob/-/models).

|Model|GPU Mem Req|Description|
|-----|-----------|-----------|
|[Anomalous Behavior Profiling (ABP)](https://github.com/nv-morpheus/Morpheus/blob/-/models#anomalous-behavior-profiling-abp)|2015MiB|This model is an example of a binary classifier to differentiate between anomalous GPU behavior such as crypto mining / GPU malware, and non-anomalous GPU-based workflows (e.g., ML/DL training). The model is an XGBoost model.|
|[Digital Fingerprinting (DFP)](https://github.com/nv-morpheus/Morpheus/blob/-/models#digital-fingerprinting-dfp)|4.97MiB|This use case is currently implemented to detect changes in users' behavior that indicate a change from a human to a machine or a machine to a human. The model is an ensemble of an Autoencoder and fast Fourier transform reconstruction.|
|[Fraud Detection](https://github.com/nv-morpheus/Morpheus/blob/-/models#fraud-detection)|76.55MiB|This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of GraphSAGE along XGBoost is used to identify frauds in the transaction networks.|
|[Flexible Log Parsing](https://github.com/nv-morpheus/Morpheus/blob/-/models#flexible-log-parsing)|1612MiB|This model is an example of using Named Entity Recognition (NER) for log parsing, specifically apache web logs.|
|[Phishing Email Detection](https://github.com/nv-morpheus/Morpheus/blob/-/models#phishing-email-detection)|1564MiB|Phishing email detection is a binary classifier differentiating between phishing/spam and non-phishing/spam emails and SMS messages.|
|[Ransomware Detection Model](https://github.com/nv-morpheus/Morpheus/blob/-/models#ransomware-detection-via-appshield)|n/a|This model shows an application of DOCA AppShield to use data from volatile memory to classify processes as ransomware or bengin. This model uses a sliding window over time and feeds derived data into a random forest classifiers of various lengths depending on the amount of data collected.|
|[Root Cause Analysis](https://github.com/nv-morpheus/Morpheus/blob/-/models#root-cause-analysis)|1583MiB|Root cause analysis is a binary classifier differentiating between ordinary logs and errors/problems/root causes in the log files.|
|[Sensitive Information Detection (SID)](https://github.com/nv-morpheus/Morpheus/blob/-/models#sensitive-information-detection-sid)|166.6MiB|SID is a classifier, designed to detect sensitive information (e.g., AWS credentials, GitHub credentials) in unencrypted data. This example model classifies text containing these 10 categories of sensitive information- address, bank account, credit card number, email address, government id number, full name, password, phone number, secret keys, and usernames.|