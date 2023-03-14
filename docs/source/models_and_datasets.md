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
|[Anomalous Behavior Profiling (ABP)](https://github.com/nv-morpheus/Morpheus/blob/-/models#anomalous-behavior-profiling-abp)|2015MiB|This model is an example of a binary classifier to differentiate between anomalous GPU behavior such as crypto mining / GPU malware, and non-anomalous GPU-based workflows (for example, ML/DL training). The model is an XGBoost model.|
|[Digital Fingerprinting (DFP)](https://github.com/nv-morpheus/Morpheus/blob/-/models#digital-fingerprinting-dfp)|4.97MiB|This use case is currently implemented to detect changes in a users' behavior that indicates a change from a human to a machine or a machine to a human. The model is an ensemble of an Autoencoder and fast Fourier transform reconstruction.||[Fraud Detection](https://github.com/nv-morpheus/Morpheus/blob/-/models#fraud-detection)|76.55MiB|This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of GraphSAGE along XGBoost is used to identify frauds in the transaction networks.|
|[Fraud Detection](https://github.com/nv-morpheus/Morpheus/blob/-/models#fraud-detection)|76.55MiB|This model shows an application of a graph neural network for fraud detection in a credit card transaction graph. A transaction dataset that includes three types of nodes, transaction, client, and merchant nodes is used for modeling. A combination of [GraphSAGE](https://snap.stanford.edu/graphsage/) along with [XGBoost](https://xgboost.readthedocs.io/en/stable/) is used to identify frauds in the transaction networks.||[Ransomware Detection Model](https://github.com/nv-morpheus/Morpheus/blob/-/models#ransomware-detection-via-appshield)|n/a|This model shows an application of DOCA AppShield to use data from volatile memory to classify processes as ransomware or bengin. This model uses a sliding window over time and feeds derived data into a random forest classifiers of various lengths depending on the amount of data collected.|
|[Flexible Log Parsing](https://github.com/nv-morpheus/Morpheus/blob/-/models#flexible-log-parsing)|1612MiB|This model is an example of using Named Entity Recognition (NER) for log parsing, specifically [Apache HTTP Server](https://httpd.apache.org/) logs.||[Ransomware Detection Model](https://github.com/nv-morpheus/Morpheus/blob/-/models#ransomware-detection-via-appshield)|n/a|This model shows an application of [DOCA App Shield](https://docs.nvidia.com/doca/sdk/app-shield-programming-guide/index.html) to use data from volatile memory to classify processes as ransomware or benign. This model uses a sliding window over time and feeds derived data into random forest classifiers of various lengths depending on the amount of data collected.|