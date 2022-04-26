# Morpheus Quickstart Guide

## Table of Contents
- [Morpheus Quickstart Guide](#morpheus-quickstart-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Overview](#overview)
    - [Features](#features)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Set up NGC API Key and Install NGC Registry CLI](#set-up-ngc-api-key-and-install-ngc-registry-cli)
    - [Create Namespace for Morpheus](#create-namespace-for-morpheus)
    - [Install Morpheus AI Engine](#install-morpheus-ai-engine)
    - [Install Morpheus SDK Client](#install-morpheus-sdk-client)
      - [Morpheus SDK Client in Sleep Mode](#morpheus-sdk-client-in-sleep-mode)
    - [Models for MLFlow Plugin Deployment](#models-for-mlflow-plugin-deployment)
    - [Install Morpheus MLFlow Triton Plugin](#install-morpheus-mlflow-triton-plugin)
    - [Model Deployment](#model-deployment)
    - [Verify Model Deployment](#verify-model-deployment)
    - [Create Kafka Topics](#create-kafka-topics)
  - [Example Workflows](#example-workflows)
    - [Run AutoEncoder Digital Fingerprinting Pipeline](#run-autoencoder-digital-fingerprinting-pipeline)
    - [Run NLP Phishing Detection Pipeline](#run-nlp-phishing-detection-pipeline)
    - [Run NLP Sensitive Information Detection Pipeline](#run-nlp-sensitive-information-detection-pipeline)
    - [Run FIL Anomalous Behavior Profiling Pipeline](#run-fil-anomalous-behavior-profiling-pipeline)
    - [Verify Running Pipeline](#verify-running-pipeline)
  - [Appendix A](#appendix-a)
    - [Prerequisites and Installation for AWS](#prerequisites-and-installation-for-aws)
      - [Prerequisites](#prerequisites-1)
      - [Install Cloud Native Core Stack for AWS](#install-cloud-native-core-stack-for-aws)
    - [Prerequisites and Installation for Ubuntu](#prerequisites-and-installation-for-ubuntu)
      - [Prerequisites](#prerequisites-2)
    - [Installing Cloud Native Core Stack on NVIDIA Certified Systems](#installing-cloud-native-core-stack-on-nvidia-certified-systems)
  - [Appendix B](#appendix-b)
    - [Kafka Topic Commands](#kafka-topic-commands)
    - [Using Morpheus SDK Client to Run Pipelines](#using-morpheus-sdk-client-to-run-pipelines)
  - [Appendix C](#appendix-c)
    - [Additional Documentation](#additional-documentation)
    - [Troubleshooting](#troubleshooting)
      - [Common Problems](#common-problems)

-   [Example Workflows](#example-workflows)
    -   [Run AutoEncoder Digital Fingerprinting Pipeline](#run-autoencoder-digital-fingerprinting-pipeline)
    -   [Run NLP Phishing Detection Pipeline](#run-nlp-phishing-detection-pipeline)
    -   [Run NLP Sensitive Information Detection Pipeline](#run-nlp-sensitive-information-detection-pipeline)
    -   [Run FIL Anomalous Behavior Profiling Pipeline](#run-fil-anomalous-behavior-profiling-pipeline)
    -   [Verify Running Pipeline](#verify-running-pipeline)

-   [Appendix A](#appendix-a)
    -   [Prerequisites and Installation for AWS](#prerequisites-and-installation-for-aws)
        -   [Prerequisites](#prerequisites-1)
        -   [Install Cloud Native Core Stack for AWS](#install-cloud-native-core-stack-for-aws)
    -   [Prerequisites and Installation for Ubuntu](#prerequisites-and-installation-for-ubuntu)
        -   [Prerequisites](#prerequisites-2)
        -   [Installing Cloud Native Core Stack on NVIDIA Certified Systems](#installing-cloud-native-core-stack-on-nvidia-certified-systems)

-   [Appendix B](#appendix-b)
    -   [Kafka Topic Commands](#kafka-topic-commands)
    -   [Using Morpheus SDK Client to Run Pipelines](#using-morpheus-sdk-client-to-run-pipelines)

-   [Appendix C](#appendix-c)
    -   [Additional Documentation](#additional-documentation)
-   [Troubleshooting](#troubleshooting)


## Introduction

This quick start guide provides the necessary instructions to set up the minimum infrastructure and configuration needed to deploy the Morpheus Developer Kit and includes example workflows leveraging the deployment.

- This quick start guide consists of the following steps:
- Set up of the NVIDIA Cloud Native Core Stack
- Set up Morpheus AI Engine
- Set up Morpheus SDK Client
- Models for MLFlow Triton Plugin Deployments
- Set up Morpheus MLFlow Triton Plugin
- Deploy models to Triton inference server
- Create Kafka topics
- Run example workloads

**Note**: This guide requires access to the NGC Public Catalog.

## Overview

Morpheus makes it easy to build and scale cybersecurity applications that harness adaptive pipelines supporting a wider range of model complexity than previously feasible. Morpheus makes it possible to analyze up to 100% of your data in real-time, for more accurate detection and faster remediation of threats as they occur. Morpheus also provides the ability to leverage AI to adjust to threats and compensate on the fly, at line rate.

NVIDIA Morpheus enables organizations to attack the issue of cybersecurity head on. Rather than continuously chasing the cybersecurity problem, Morpheus provides the ability to propel you ahead of the breach and address the cybersecurity issue. With the world in a "discover and respond" state, where companies are finding breaches much too late, in a way that is way behind the curve, NVIDIA’s Morpheus cybersecurity AI framework enables any organization to warp to the present and begin to defend itself in real time.

The Morpheus Developer Kit allows developers to quickly and easily set up example pipelines to run inference on different sample models provided from NVIDIA and experiment with the features and capabilities available within the Morpheus framework to address their cybersecurity and information security use cases.

### Features

- **Built on RAPIDS™**

    Built on the RAPIDS™ libraries, deep learning frameworks, and NVIDIA Triton™ Inference Server, Morpheus simplifies the analysis of logs and telemetry to help detect and mitigate security threats.

- **Massive Performance and Scale**

    Enables AI inference and real-time monitoring of every server and packet across the entire network.

- **Rapid Development and Deployment**

    Integrates AI frameworks and tools that make it easier for developers to build cybersecurity solutions. Organizations that lack AI expertise can still leverage AI for cybersecurity because Morpheus leverages tools for every stage of the AI workflow, from data preparation to training, inference, and deploying at scale. 

- **Real-time Telemetry**

    The Morpheus native graph streaming engine can receive rich, real-time network telemetry from every NVIDIA BlueField DPU-accelerated server or NVIDIA AppShield in the data center without impacting performance. Integrating the framework into a third-party cybersecurity offering brings the world’s best AI computing to communication networks. 

- **AI Cybersecurity Capabilities**

    Deploy your own models using common deep learning frameworks. Or get a jump-start in building applications to identify leaked sensitive information, detect malware or fraud, do network mapping, flag user behavior changes, or and identify errors via logs by using one of NVIDIA’s pre-trained and tested models.

## Setup

### Prerequisites
1.  Refer to [Appendix A](#appendix-a) for Cloud (AWS) or On-Prem (Ubuntu)
2.  Registration in the NGC Public Catalog

Continue with the setup steps below once the host system is installed and configured and satisfies all prerequisites.

### Set up NGC API Key and Install NGC Registry CLI

First, you will need to set up your NGC API Key to access all the Morpheus components, using the linked instructions from the [NGC Registry CLI User Guide].

Once you’ve created your API key, create an environment variable containing your API key for use by the commands used further in this document:

```bash
$ export API_KEY="<NGC_API_KEY>"
```

Next, install and configure the NGC Registry CLI on your system using the linked instructions from the [NGC Registry CLI User Guide].

### Create Namespace for Morpheus

Next, create a namespace and an environment variable for the namespace to organize the Kubernetes cluster deployed via the Cloud Native Core Stack and logically separate Morpheus related deployments from other projects using the following command:

```bash
$ kubectl create namespace <YOUR_NAMESPACE>
$ export NAMESPACE="<YOUR_NAMESPACE>"
```

### Install Morpheus AI Engine

The Morpheus AI Engine consists of the following components:
-   Triton Inference Server [ **ai-engine** ] from NVIDIA for processing inference requests.
-   Kafka Broker [ **broker** ] to consume and publish messages.
-   Zookeeper [ **zookeeper** ] to maintain coordination between the Kafka Brokers.

Follow the below steps to install Morpheus AI Engine:

```bash
$ helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-ai-engine-22.04.tgz --username='$oauthtoken' --password=$API_KEY --untar
```
```bash
$ helm install --set ngc.apiKey="$API_KEY" \
             --namespace $NAMESPACE \
             <YOUR_RELEASE_NAME> \
             morpheus-ai-engine
```

After the installation, You can verify that the Kubernetes pods are running successfully using the following command:

```bash
$ kubectl -n $NAMESPACE get all
```

Output:

```console
pod/ai-engine-65f59ddcf7-mdmdt   1/1     Running   0          54s
pod/broker-76f7c64dc9-6rldp      1/1     Running   1          54s
pod/zookeeper-87f9f4dd-znjnb     1/1     Running   0          54s

NAME                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
service/ai-engine   ClusterIP   10.109.56.231    <none>        8000/TCP,8001/TCP,8002/TCP   54s
service/broker      ClusterIP   10.101.103.250   <none>        9092/TCP                     54s
service/zookeeper   ClusterIP   10.110.55.141    <none>        2181/TCP                     54s

NAME                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/ai-engine   1/1     1            1           54s
deployment.apps/broker      1/1     1            1           54s
deployment.apps/zookeeper   1/1     1            1           54s

NAME                                   DESIRED   CURRENT   READY   AGE
replicaset.apps/ai-engine-65f59ddcf7   1         1         1       54s
replicaset.apps/broker-76f7c64dc9      1         1         1       54s
replicaset.apps/zookeeper-87f9f4dd     1         1         1       54s
```

### Install Morpheus SDK Client
Run the following commands to pull and install the Morpheus CLI on your instance:

```bash
$ helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-sdk-client-22.04.tgz --username='$oauthtoken' --password=$API_KEY --untar
```

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
               --set sdk.args="<YOUR_WORKFLOW_RUN_COMMAND>" \
               --namespace $NAMESPACE \
               <YOUR_RELEASE_NAME> \
               morpheus-sdk-client
```

**Note**: The install command references the run pipeline command argument, provided in the [example workflows](#example-workflows) below.

#### Morpheus SDK Client in Sleep Mode
Using the default `sdk.args` from the charts, Morpheus SDK Client would be put into sleep mode.

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
               --namespace $NAMESPACE \
               helper \
               morpheus-sdk-client
```

Check the status of the pod to make sure it's up and running.

```bash
$ kubectl -n $NAMESPACE get all | grep sdk-cli-helper
```

Output:

```console
pod/sdk-cli-helper           1/1     Running   0               41s
```

### Models for MLFlow Plugin Deployment

Connect to the **sdk-cli-helper** and copy models to `/common`, which is mapped to `/opt/morpheus/common` on the host and where MLFlow will have access to model files.

```bash
$ kubectl -n $NAMESPACE exec sdk-cli-helper -- cp -R /workspace/models /common
```

### Install Morpheus MLFlow Triton Plugin

The Morpheus MLFlow Triton Plugin is used to deploy, update, and remove models from the Morpheus AI Engine. MLFlow server UI can be accessed using NodePort 30500
Follow the below steps to install Morpheus MlFLow Triton Plugin:

```bash
$ helm fetch https://helm.ngc.nvidia.com/nvidia/morpheus/charts/morpheus-mlflow-22.04.tgz --username='$oauthtoken' --password=$API_KEY --untar
```
```bash
$ helm install --set ngc.apiKey="$API_KEY" \
             --namespace $NAMESPACE \
             <YOUR_RELEASE_NAME> \
             morpheus-mlflow
```

**Note**: If the default port is already allocated, helm throws below error. Choose an alternative by adjusting the `dashboardPort` value in the `morpheus-mlflow/values.yaml` file, remove the previous release and reinstall it.

```console
Error: Service "mlflow" is invalid: spec.ports[0].nodePort: Invalid value: 30500: provided port is already allocated
```

After the installation, you can verify that the MLFlow pod is running successfully using the following command:

```bash
$ kubectl -n $NAMESPACE get all | grep  pod/mlflow
```

Output:
```bash
pod/mlflow-6d98        1/1     Running   0          39s
```

### Model Deployment
Attach to the MLFLow pod to publish models to MLFlow server and then deploy it onto Morpheus AI Engine:

```bash
$ kubectl -n $NAMESPACE exec -it deploy/mlflow -- bash
```

```console
(mlflow) root@mlflow-6d98:/mlflow#
```

`Important`: When (mlflow) is present, commands are directly within the container.

Let's have a look at how to use the MLFlow Triton plugin before we start deploying models.
Publish models to MLFlow server:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name <REF_MODEL_NAME> \
      --model_directory <MODEL_DIR_PATH> \
      --flavor <MODEL_FLAVOR>
```

Deploy models to Morpheus AI Engine:

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
     --flavor <MODEL_FLAVOR> \
     --name <REF_MODEL_NAME> \
     -m models:/<REF_MODEL_NAME>/1 \
     -C "version=<VERSION_NUMBER>"
```

Update deployed models in Morpheus AI Engine:

```
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments update -t triton \
     --flavor <MODEL_FLAVOR> \
     --name <REF_MODEL_NAME>/<EXISTING_VERSION_NUMBER> \
     -m models:/<REF_MODEL_NAME>/<DESIRED_VERSION_NUMBER>
```

Delete deployed models from Morpheus AI Engine:

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments delete -t triton \
     --name <REF_MODEL_NAME>/<VERSION_NUMBER>
```

Now that we've figured out how to deploy models let's move on to the next step. Now it's time to deploy the relevant models, which have already been copied to `/opt/morpheus/common/models` which are bound to `/common/models` within the MLFlow pod.

```bash
(mlflow) root@mlflow-6d98:/mlflow# ls -lrt /common/models
```

Output:

```console
drwxr-xr-x 3 ubuntu ubuntu 4096 Apr 13 23:47 sid-minibert-onnx
drwxr-xr-x 2 root   root   4096 Apr 21 17:09 abp-models
drwxr-xr-x 4 root   root   4096 Apr 21 17:09 datasets
drwxr-xr-x 4 root   root   4096 Apr 21 17:09 fraud-detection-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:09 hammah-models
drwxr-xr-x 3 root   root   4096 Apr 21 17:10 mlflow
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 log-parsing-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 phishing-models
drwxr-xr-x 2 root   root   4096 Apr 21 17:10 sid-models
drwxr-xr-x 8 root   root   4096 Apr 21 17:10 training-tuning-scripts
drwxr-xr-x 7 root   root   4096 Apr 21 17:10 validation-inference-scripts
drwxr-xr-x 7 root   root   4096 Apr 21 17:10 triton-model-repo
-rw-r--r-- 1 root   root   4213 Apr 21 17:10 README.md
-rw-r--r-- 1 root   root   4862 Apr 21 17:10 model_cards.csv
-rw-r--r-- 1 root   root   1367 Apr 21 17:10 model-information.csv
```


Publish and deploy sid-minibert-onnx model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name sid-minibert-onnx \
      --model_directory /common/models/triton-model-repo/sid-minibert-onnx \
      --flavor triton
```

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name sid-minibert-onnx \
      -m models:/sid-minibert-onnx/1 \
      -C "version=1"
```

Publish and deploy phishing-bert-onnx model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name phishing-bert-onnx \
      --model_directory /common/models/triton-model-repo/phishing-bert-onnx \
      --flavor triton
```
```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name phishing-bert-onnx \
      -m models:/phishing-bert-onnx/1 \
      -C "version=1"
```

Publish and deploy abp-nvsmi-xgb model:

```bash
(mlflow) root@mlflow-6d98:/mlflow# python publish_model_to_mlflow.py \
      --model_name abp-nvsmi-xgb \
      --model_directory /common/models/triton-model-repo/abp-nvsmi-xgb \
      --flavor triton
```

```bash
(mlflow) root@mlflow-6d98:/mlflow# mlflow deployments create -t triton \
      --flavor triton \
      --name abp-nvsmi-xgb \
      -m models:/abp-nvsmi-xgb/1 \
      -C "version=1"
```

Exit from the container

```bash
(mlflow) root@mlflow-6d98:/mlflow# exit
```

### Verify Model Deployment
Run the following command to verify that the models were successfully deployed on the AI Engine:

```bash
$ kubectl -n $NAMESPACE logs deploy/ai-engine
```

Output:
```console
I1202 14:09:03.098085 1 api.cu:79] TRITONBACKEND_ModelInitialize: abp-nvsmi-xgb (version 1)
I1202 14:09:03.101910 1 api.cu:123] TRITONBACKEND_ModelInstanceInitialize: abp-nvsmi-xgb_0 (GPU device 0)
I1202 14:09:03.543719 1 model_instance_state.cu:101] Using GPU for predicting with model 'abp-nvsmi-xgb_0'
I1202 14:09:03.563425 1 api.cu:123] TRITONBACKEND_ModelInstanceInitialize: abp-nvsmi-xgb_0 (GPU device 1)
I1202 14:09:03.980621 1 model_instance_state.cu:101] Using GPU for predicting with model 'abp-nvsmi-xgb_0'
I1202 14:09:03.981678 1 model_repository_manager.cc:1183] successfully loaded 'abp-nvsmi-xgb' version 1
```

### Create Kafka Topics
We will need to create Kafka topics for input and output data to run some of the pipeline examples.

Check if any Kafka topics exist already. If any exist, you can either delete the previous topics or re-use them.

```bash
$ kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh --list  --zookeeper zookeeper:2181
```

Run the following command twice, once to create an input topic, and again to create an output topic, making sure that the input topic and output topic have different names:

```bash
$ kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --create \
      --bootstrap-server broker:9092 \
      --replication-factor 1 \
      --partitions 3 \
      --topic <YOUR_KAFKA_TOPIC>
```

## Example Workflows

This section describes example workflows to run on Morpheus. Four sample pipelines are provided.

1. AutoEncoder pipeline performing Human as Machine & Machine as Human (HAMMAH).
2. NLP pipeline performing Phishing Detection (PD).
3. NLP pipeline performing Sensitive Information Detection (SID).
4. FIL pipeline performing Anomalous Behavior Profiling (ABP).

Multiple command options are given for each pipeline, with varying data input/output methods, ranging from local files to Kafka Topics.

We recommend only deploying one pipeline at a time. To remove previously deployed pipelines, run the following command:

```bash
$ helm delete -n $NAMESPACE <YOUR_RELEASE_NAME>
```

To publish messages to a Kafka topic, we need to copy datasets to locations where they can be accessed from the host.

```bash
kubectl -n $NAMESPACE exec sdk-cli-helper -- cp -R /workspace/data /common
```

Refer to the Using Morpheus SDK Client to Run Pipelines section of the Appendix for more information regarding the commands.

**Note**: Before running the example pipelines, ensure that the criteria below are met:
-   Ensure that models specific to the pipeline are deployed.
-   Input and Output Kafka topics have been created.
-   Recommended to create an output directory under  `/opt/morpheus/common/data` which is bound to `/common/data` (pod/container) for storing inference or validation results.
-   Replace **<YOUR_OUTPUT_DIR>** with your directory name.
-   Replace **<YOUR_INPUT_KAFKA_TOPIC>** with your input Kafka topic name.
-   Replace **<YOUR_OUTPUT_KAFKA_TOPIC>** with your output Kafka topic name.
-   Replace **<YOUR_RELEASE_NAME>** with the name you want.


### Run AutoEncoder Digital Fingerprinting Pipeline
The following AutoEncoder pipeline example shows how to train and validate the AutoEncoder model and write the inference results to a specified location. Digital fingerprinting has also been referred to as **HAMMAH (Human as Machine <> Machine as Human)**.
These use cases are currently implemented to detect user behavior changes that indicate a change from a human to a machine or a machine to a human. The model is an ensemble of an autoencoder and fast fourier transform reconstruction.

Filter userid *(role-g)* entries and train, validate and then inference:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
    --num_threads=2 \
    --edge_buffer_size=4 \
    --pipeline_batch_size=1024 \
    --model_max_batch_size=1024 \
    --use_cpp=False \
    pipeline-ae \
      --userid_filter=role-g \
      --userid_column_name=userIdentitysessionContextsessionIssueruserName \
      from-cloudtrail --input_glob=/common/models/datasets/validation-data/hammah-*.csv \
      train-ae --train_data_glob=/common/models/datasets/training-data/hammah-*.csv \
        --seed 42 \
      preprocess \
      inf-pytorch \
      add-scores \
      timeseries --resolution=10m --zscore_threshold=8.0 \
      monitor --description 'Inference Rate' --smoothing=0.001 --unit inf \
      validate --val_file_name=/common/models/datasets/validation-data/hammah-role-g-validation-data.csv \
        --results_file_name=/common/data/<YOUR_OUTPUT_DIR>/val_hammah-role-g-pytorch-results.json \
        --index_col=_index_ \
        --exclude event_dt \
        --rel_tol=0.15 \
        --overwrite \
      serialize \
      to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/val_hammah-role-g-pytorch.csv --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Inference and training based on a userid (`user123`). The model is trained once and inference is conducted on the supplied input entries in the example pipeline below. The `--train_data_glob` parameter must be removed for continuous training.

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --num_threads=2 \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=1024 \
      --use_cpp=True \
      pipeline-ae \
        --userid_filter=user123 \
        --userid_column_name=userIdentitysessionContextsessionIssueruserName \
        from-cloudtrail --input_glob=/common/models/datasets/validation-data/hammah-*.csv \
        train-ae --train_data_glob=/common/models/datasets/training-data/hammah-*.csv \
          --seed 42 \
        preprocess \
        inf-pytorch \
        add-scores \
        timeseries --resolution=1m --zscore_threshold=8.0 --hot_start \
        monitor --description 'Inference Rate' --smoothing=0.001 --unit inf \
        serialize \
        to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/val_hammah-user123-pytorch.csv --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

### Run NLP Phishing Detection Pipeline

The following Phishing Detection pipeline examples use a pre-trained NLP model to analyze emails (body) and determine phishing or benign. Here is the sample data as shown below is used to pass as an input to the pipeline.

```json
{"data":"Abedin Huma <AbedinH@state.gov>Wednesday July 15 2009 1:44 PMRe: ArtWill be off campus at meetingBut you should definitely come I think they have found some good things."}
{"data":"See NIMills Cheryl D <MillsCD@state.gov>Saturday December 112010 1:36 PMFw: S is calling Leahy today - thx for all the help; advise if a diff no for him today"}
{"data":"Here is Draft"}
{"data":"Ok"}
```

Pipeline example to read data from a file, run inference using a `phishing-bert-onnx` model, and write inference results to the specified output file:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --num_threads=2 \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      --use_cpp=True \
      pipeline-nlp \
        --model_seq_length=128 \
        --labels_file=./data/labels_phishing.txt \
        from-file --filename=./data/email.jsonlines \
        monitor --description 'FromFile Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=./data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description 'Preprocess Rate' \
        inf-triton --model_name=phishing-bert-onnx --server_url=ai-engine:8001 --force_convert_inputs=True \
        monitor --description 'Inference Rate' --smoothing=0.001 --unit inf \
        add-class --label=pred --threshold=0.7 \
        serialize \
        to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/phishing-bert-onnx-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

When the pipeline runs successfully, an output file `phishing-bert-onnx-output.jsonlines` will appear in the output directory.

Pipeline example to read messages from an input Kafka topic, run inference using a   `phishing-bert-onnx`  model, and write the results of the inference to an output Kafka topic:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --num_threads=2 \
      --edge_buffer_size=4 \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      --use_cpp=True \
      pipeline-nlp \
        --model_seq_length=128 \
        --labels_file=./data/labels_phishing.txt \
        from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
        monitor --description 'FromKafka Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=./data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description 'Preprocess Rate' \
        inf-triton --force_convert_inputs=True --model_name=phishing-bert-onnx --server_url=ai-engine:8001 \
        monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
        add-class --label=pred --threshold=0.7 \
        serialize --exclude '^ts_' \
        to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
$ kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: /opt/morpheus/common/data/email.jsonlines>
```

**Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Run NLP Sensitive Information Detection Pipeline
The following Sensitive Information Detection pipeline examples use a pre-trained NLP model to ingest and analyze PCAP (packet capture network traffic) input sample data, like the example below, to inspect IP traffic across data center networks.

```json
{"timestamp": 1616380971990, "host_ip": "10.188.40.56", "data_len": "309", "data": "POST /simpledatagen/ HTTP/1.1\r\nHost: echo.gtc1.netqdev.cumulusnetworks.com\r\nUser-Agent: python-requests/2.22.0\r\nAccept-Encoding: gzip, deflate\r\nAccept: */*\r\nConnection: keep-alive\r\nContent-Length: 73\r\nContent-Type: application/json\r\n\r\n", "src_mac": "04:3f:72:bf:af:74", "dest_mac": "b4:a9:fc:3c:46:f8", "protocol": "6", "src_ip": "10.20.16.248", "dest_ip": "10.244.0.59", "src_port": "50410", "dest_port": "80", "flags": "24", "is_pii": false}
{"timestamp": 1616380971991, "host_ip": "10.188.40.56", "data_len": "139", "data": "\"{\\\"markerEmail\\\": \\\"FuRLFaAZ identify benefit BneiMvCZ join 92694759\\\"}\"", "src_mac": "04:3f:72:bf:af:74", "dest_mac": "b4:a9:fc:3c:46:f8", "protocol": "6", "src_ip": "10.244.0.1", "dest_ip": "10.244.0.25", "src_port": "50410", "dest_port": "80", "flags": "24", "is_pii": false}
```

Pipeline example to read data from a file, run inference using a `sid-minibert-onnx` model, and write inference results to the specified output file:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
      --num_threads=3 \
      --edge_buffer_size=4 \
      --use_cpp=True \
      --pipeline_batch_size=1024 \
      --model_max_batch_size=32 \
      pipeline-nlp \
        --model_seq_length=256 \
        from-file --filename=./data/pcap_dump.jsonlines \
        monitor --description 'FromFile Rate' --smoothing=0.001 \
        deserialize \
        preprocess --vocab_hash_file=./data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
        monitor --description='Preprocessing rate' \
        inf-triton --force_convert_inputs=True --model_name=sid-minibert-onnx --server_url=ai-engine:8001 \
        monitor --description='Inference rate' --smoothing=0.001 --unit inf \
        add-class \
        serialize --exclude '^ts_' \
        to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/sid-minibert-onnx-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

When the pipeline runs successfully, an output file *sid-minibert-onnx-output.jsonlines* will appear in the output directory.

Pipeline example to read messages from an input Kafka topic, run inference using a   *sid-minibert-onnx* model, and write the results of the inference to an output Kafka topic:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --num_threads=3 \
        --edge_buffer_size=4 \
        --use_cpp=True \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=32 \
        pipeline-nlp \
          --model_seq_length=256 \
          from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
          monitor --description 'FromKafka Rate' --smoothing=0.001 \
          deserialize \
          preprocess --vocab_hash_file=./data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
          monitor --description='Preprocessing Rate' \
          inf-triton --force_convert_inputs=True --model_name=sid-minibert-onnx --server_url=ai-engine:8001 \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^ts_' \
          to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
$ kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: ${HOME}/data/pcap_dump.jsonlines>
```

**Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Run FIL Anomalous Behavior Profiling Pipeline
The following Anomalous Behavior Profiling pipeline examples use a pre-trained FIL model to ingest and analyze Nvidia System Management Interface (nvidia-smi) logs, like the example below, as input sample data to identify crypto mining activity on GPU devices.

```json
{"nvidia_smi_log.gpu.pci.tx_util": "0 KB/s", "nvidia_smi_log.gpu.pci.rx_util": "0 KB/s", "nvidia_smi_log.gpu.fb_memory_usage.used": "3980 MiB", "nvidia_smi_log.gpu.fb_memory_usage.free": "12180 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.total": "16384 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.used": "11 MiB", "nvidia_smi_log.gpu.bar1_memory_usage.free": "16373 MiB", "nvidia_smi_log.gpu.utilization.gpu_util": "0 %", "nvidia_smi_log.gpu.utilization.memory_util": "0 %", "nvidia_smi_log.gpu.temperature.gpu_temp": "61 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold": "90 C", "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold": "87 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold": "83 C", "nvidia_smi_log.gpu.temperature.memory_temp": "57 C", "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold": "85 C", "nvidia_smi_log.gpu.power_readings.power_draw": "61.77 W", "nvidia_smi_log.gpu.clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.clocks.sm_clock": "1530 MHz", "nvidia_smi_log.gpu.clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.clocks.video_clock": "1372 MHz", "nvidia_smi_log.gpu.applications_clocks.graphics_clock": "1312 MHz", "nvidia_smi_log.gpu.applications_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock": "1312 MHz", "nvidia_smi_log.gpu.default_applications_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.max_clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.max_clocks.sm_clock": "1530 MHz", "nvidia_smi_log.gpu.max_clocks.mem_clock": "877 MHz", "nvidia_smi_log.gpu.max_clocks.video_clock": "1372 MHz", "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock": "1530 MHz", "nvidia_smi_log.gpu.processes.process_info.0.process_name": "python", "nvidia_smi_log.gpu.processes.process_info.1.process_name": "tritonserver", "hostname": "ip-10-100-8-98", "timestamp": 1615542360.9566503}
```

Pipeline example to read data from a file, run inference using an `abp-nvsmi-xgb` model, and write inference results to the specified output file.

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --num_threads=3 \
        --edge_buffer_size=4 \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=64 \
        --use_cpp=True \
        pipeline-fil \
          from-file --filename=./data/nvsmi.jsonlines \
          monitor --description 'FromFile Rate' --smoothing=0.001 \
          deserialize \
          preprocess \
          monitor --description='Preprocessing Rate' \
          inf-triton --model_name=abp-nvsmi-xgb --server_url=ai-engine:8001 --force_convert_inputs=True \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^nvidia_smi_log' --exclude '^ts_' \
          to-file --filename=/common/data/<YOUR_OUTPUT_DIR>/abp-nvsmi-xgb-output.jsonlines --overwrite" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Pipeline example to read messages from an input Kafka topic, run inference using an `abp-nvsmi-xgb` model, and write the results of the inference to an output Kafka topic:

```bash
$ helm install --set ngc.apiKey="$API_KEY" \
    --set sdk.args="morpheus --log_level=DEBUG run \
        --num_threads=3 \
        --pipeline_batch_size=1024 \
        --model_max_batch_size=64 \
        --use_cpp=True \
        pipeline-fil \
          from-kafka --input_topic <YOUR_INPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092 \
          monitor --description 'FromKafka Rate' --smoothing=0.001 \
          deserialize \
          preprocess \
          monitor --description='Preprocessing Rate' \
          inf-triton --model_name=abp-nvsmi-xgb --server_url=ai-engine:8001 --force_convert_inputs=True \
          monitor --description='Inference Rate' --smoothing=0.001 --unit inf \
          add-class \
          serialize --exclude '^nvidia_smi_log' \ --exclude '^ts_' \
          to-kafka --output_topic <YOUR_OUTPUT_KAFKA_TOPIC> --bootstrap_servers broker:9092" \
    --namespace $NAMESPACE \
    <YOUR_RELEASE_NAME> \
    morpheus-sdk-client
```

Make sure you create input and output Kafka topics before you start the pipeline. After the pipeline has been started, load the individual corresponding data files from the downloaded sample into the selected input topic using the command below:

```bash
$ kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_INPUT_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE_PATH_EXAMPLE: ${HOME}/data/nvsmi.jsonlines>
```

**Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.

### Verify Running Pipeline
Once you’ve deployed the SDK client to run a pipeline, you can check the status of the pod using the following command:

```bash
$ kubectl -n $NAMESPACE get pods sdk-cli-<RELEASE_NAME>
NAME                       READY   STATUS    RESTARTS   AGE
sdk-cli-6c9575f648-gfdd2   1/1     Running   0          3m23s
```

Then check that the pipeline is running successfully using the following command:

```bash
kubectl -n $NAMESPACE logs sdk-cli-<RELEASE_NAME>
```

Output:

```console
Configuring Pipeline via CLI
Starting pipeline via CLI... Ctrl+C to Quit
Preprocessing rate: 7051messages [00:09, 4372.75messages/s]
Inference rate: 7051messages [00:04, 4639.40messages/s]
```

## Appendix A

### Prerequisites and Installation for AWS

#### Prerequisites
1.  AWS account with the ability to create/modify EC2 instances
2.  AWS EC2 G4 instance with T4 or V100 GPU, at least 64GB RAM, 8 cores CPU, and 100 GB storage.

#### Install Cloud Native Core Stack for AWS
On your AWS EC2 G4 instance, follow the instructions in the linked document to install [NVIDIA’s Cloud Native Core Stack for AWS][NVIDIA’s Cloud Native Core Stack].

### Prerequisites and Installation for Ubuntu

#### Prerequisites
1.  NVIDIA-Certified System
2.  NVIDIA Pascal GPU or newer
3.  Ubuntu 20.04 LTS or newer

### Installing Cloud Native Core Stack on NVIDIA Certified Systems
On your NVIDIA-Certified System, follow the instructions in the linked document to install [NVIDIA’s Cloud Native Core Stack].

## Appendix B

### Kafka Topic Commands

List available Kafka topics.

```bash
$ kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --list  --zookeeper zookeeper:2181
```

Create a partitioned Kafka topic with a single replication factor.

```bash
$ kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --create \
      --bootstrap-server broker:9092 \
      --replication-factor 1 \
      --partitions 1 \
      --topic <YOUR_KAFKA_TOPIC>
```

Load data from a file to Kafka topic:

```bash
$ kubectl -n $NAMESPACE exec -it deploy/broker -c broker -- kafka-console-producer.sh \
       --broker-list broker:9092 \
       --topic <YOUR_KAFKA_TOPIC> < \
       <YOUR_INPUT_DATA_FILE>
```

**Note**: This should be used for development purposes only via this developer kit. Loading from the file into Kafka should not be used in production deployments of Morpheus.


Consume messages from Kafka topic:

```bash
$ kubectl -n $NAMESPACE exec deploy/broker  -c broker -- kafka-console-consumer.sh \
       --bootstrap-server broker:9092 \
       --topic <YOUR_KAFKA_TOPIC> \
       --group <YOUR_CONSUMER_GROUP_ID>
```

Delete Kafka topic:

```bash
$ kubectl -n $NAMESPACE exec deploy/broker -c broker -- kafka-topics.sh \
      --delete --zookeeper zookeeper:2181 \
      --topic <YOUR_KAFKA_TOPIC>
```

### Using Morpheus SDK Client to Run Pipelines

The Morpheus SDK client allows you to configure several supported pipelines and provides flexibility to execute the pipeline in multithread mode.

```bash
(morpheus) root@sdk-cli:/workspace# morpheus run --help
```
```console
Usage: morpheus run [OPTIONS] COMMAND [ARGS]...

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
                                  [default: 4; x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers  [default: 256;
                                  x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model
                                  [default: 8; x>=1]
  --edge_buffer_size INTEGER RANGE
                                  The size of buffered channels to use between
                                  nodes in a pipeline. Larger values reduce
                                  backpressure at the cost of memory. Smaller
                                  values will push messages through the
                                  pipeline quicker. Must be greater than 1 and
                                  a power of 2 (i.e. 2, 4, 8, 16, etc.)
                                  [default: 128; x>=2]
  --use_cpp BOOLEAN               Whether or not to use C++ node and message
                                  types or to prefer python. Only use as a
                                  last resort if bugs are encountered
                                  [default: True]
  --help                          Show this message and exit.

Commands:
  pipeline-ae   Run the inference pipeline with an AutoEncoder model
  pipeline-fil  Run the inference pipeline with a FIL model
  pipeline-nlp  Run the inference pipeline with a NLP model
```

Three different pipelines are currently supported, a pipeline running an NLP model, a pipeline running a FIL model, and a pipeline running an AutoEncoder model.


The Morpheus SDK Client provides the commands below to run the NLP pipeline:

```bash
(morpheus) root@sdk-cli:/workspace# morpheus run pipeline-nlp --help
```

```console
Usage: morpheus run pipeline-nlp [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                 [ARGS]...]...

  Configure and run the pipeline. To configure the pipeline, list the stages
  in the order that data should flow. The output of each stage will become the
  input for the next stage. For example, to read, classify and write to a
  file, the following stages could be used

  pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
  --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

  Pipelines must follow a few rules:
  1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
  2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
  3. Only one inference stage can be used. Zero is also fine
  4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

Options:
  --model_seq_length INTEGER RANGE
                                  Limits the length of the sequence returned.
                                  If tokenized string is shorter than
                                  max_length, output will be padded with 0s.
                                  If the tokenized string is longer than
                                  max_length and do_truncate == False, there
                                  will be multiple returned sequences
                                  containing the overflowing token-ids.
                                  Default value is 256  [default: 256; x>=1]
  --labels_file FILE              Specifies a file to read labels from in
                                  order to convert class IDs into labels. A
                                  label file is a simple text file where each
                                  line corresponds to a label  [default:
                                  data/labels_nlp.txt]
  --viz_file FILE                 Save a visualization of the pipeline at the
                                  specified location
  --help                          Show this message and exit.

Commands:
  add-class     Add detected classifications to each message
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  gen-viz       (Deprecated) Write out vizualization data frames
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Include & exclude columns from messages
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```

Morpheus SDK Client provides the commands below to run the FIL pipeline:

```bash
(morpheus) root@sdk-cli:/workspace# morpheus run pipeline-fil --help
```

```console
Usage: morpheus run pipeline-fil [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                 [ARGS]...]...

  Configure and run the pipeline. To configure the pipeline, list the stages
  in the order that data should flow. The output of each stage will become the
  input for the next stage. For example, to read, classify and write to a
  file, the following stages could be used

  pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
  --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

  Pipelines must follow a few rules:
  1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
  2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
  3. Only one inference stage can be used. Zero is also fine
  4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

Options:
  --model_fea_length INTEGER RANGE
                                  Number of features trained in the model
                                  [default: 29; x>=1]
  --labels_file FILE              Specifies a file to read labels from in
                                  order to convert class IDs into labels. A
                                  label file is a simple text file where each
                                  line corresponds to a label. If unspecified,
                                  only a single output label is created for
                                  FIL
  --columns_file FILE             Specifies a file to read column features.
                                  [default: data/columns_fil.txt]
  --viz_file FILE                 Save a visualization of the pipeline at the
                                  specified location
  --help                          Show this message and exit.

Commands:
  add-class     Add detected classifications to each message
  add-scores    Add probability scores to each message
  buffer        (Deprecated) Buffer results
  delay         (Deprecated) Delay results for a certain duration
  deserialize   Deserialize source data from JSON.
  dropna        Drop null data entries from a DataFrame
  filter        Filter message by a classification threshold
  from-file     Load messages from a file
  from-kafka    Load messages from a Kafka cluster
  inf-identity  Perform a no-op inference for testing
  inf-pytorch   Perform inference with PyTorch
  inf-triton    Perform inference with Triton
  mlflow-drift  Report model drift statistics to ML Flow
  monitor       Display throughput numbers at a specific point in the pipeline
  preprocess    Convert messages to tokens
  serialize     Include & exclude columns from messages
  to-file       Write all messages to a file
  to-kafka      Write all messages to a Kafka cluster
  validate      Validates pipeline output against an expected output
```

Morpheus SDK Client provides the commands below to run the AutoEncoder pipeline:

```bash
(morpheus) root@sdk-cli:/workspace# morpheus run pipeline-ae --help
```

```console
Usage: morpheus run pipeline-ae [OPTIONS] COMMAND1 [ARGS]... [COMMAND2
                                [ARGS]...]...

  Configure and run the pipeline. To configure the pipeline, list the stages
  in the order that data should flow. The output of each stage will become the
  input for the next stage. For example, to read, classify and write to a
  file, the following stages could be used

  pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
  --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

  Pipelines must follow a few rules:
  1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
  2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
  3. Only one inference stage can be used. Zero is also fine
  4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

Options:
  --columns_file FILE        [default: data/columns_ae.txt]
  --labels_file FILE         Specifies a file to read labels from in order to
                             convert class IDs into labels. A label file is a
                             simple text file where each line corresponds to a
                             label. If unspecified, only a single output label
                             is created for FIL
  --userid_column_name TEXT  Which column to use as the User ID.  [default:
                             userIdentityaccountId; required]
  --userid_filter TEXT       Specifying this value will filter all incoming
                             data to only use rows with matching User IDs.
                             Which column is used for the User ID is specified
                             by `userid_column_name`
  --viz_file FILE            Save a visualization of the pipeline at the
                             specified location
  --help                     Show this message and exit.

Commands:
  add-class        Add detected classifications to each message
  add-scores       Add probability scores to each message
  buffer           (Deprecated) Buffer results
  delay            (Deprecated) Delay results for a certain duration
  filter           Filter message by a classification threshold
  from-cloudtrail  Load messages from a Cloudtrail directory
  gen-viz          (Deprecated) Write out vizualization data frames
  inf-pytorch      Perform inference with PyTorch
  inf-triton       Perform inference with Triton
  monitor          Display throughput numbers at a specific point in the
                   pipeline
  preprocess       Convert messages to tokens
  serialize        Include & exclude columns from messages
  timeseries       Perform time series anomaly detection and add prediction.
  to-file          Write all messages to a file
  to-kafka         Write all messages to a Kafka cluster
  train-ae         Deserialize source data from JSON
  validate         Validates pipeline output against an expected output
```

## Appendix C

### Additional Documentation
For more information on how to use the Morpheus CLI to customize and run your own optimized AI pipelines, Refer to below documentation.
- [Morpheus Contribution]
- [Morpheus Developer Guide]
- [Morpheus Pipeline Examples]


### Troubleshooting
This section lists solutions to problems you might encounter with Morpheus or from it's supporting components.

#### Common Problems

- Models Unloaded After Reboot
  - When the pod is restarted, K8s will not automatically load the models. Since models are deployed to *ai-engine* in explicit mode using MLFlow, we'd have to manually deploy them again using the [Model Deployment](#model-deployment) process.
- AI Engine CPU Only Mode
  - After a server restart, the ai-engine pod on k8s can start up before the gpu operator infrastructure is available, making it "think" there is no driver installed (i.e., CPU -only mode).
- Improve Pipeline Message Processing Rate
  - Below settings need to be considered
    - Provide the workflow with the optimal number of threads (`—num threads`), as having more or fewer threads can have an impact on pipeline performance.
    - Consider adjusting `pipeline_batch_size` and `model_max_batch_size`
- Kafka Message Offset Commit Fail
  - Error Message
  ```console
  1649207839.253|COMMITFAIL|rdkafka#consumer-2| [thrd:main]: Offset commit (manual) failed for 1/1 partition(s) in join-state wait-unassign-call: Broker: Unknown member: topic[0]@112071(Broker: Unknown member)
  ```
  - Problem: If the standalone kafka cluster is receiving significant message throughput from the producer, this error may happen.

  - Solution: Reinstall the Morpheus workflow and reduce the Kafka topic's message retention time and message producing rate.


<!---
## Known Issues

| Issue | Description |
| ------ | ------ |
| | |

Let's add any important issues that need to be brought to the attention of users here.
-->

[Morpheus Pipeline Examples]: https://github.com/NVIDIA/Morpheus/tree/branch-22.04/examples
[Morpheus Contribution]: https://github.com/NVIDIA/Morpheus/blob/branch-22.04/CONTRIBUTING.md
[Morpheus Developer Guide]: https://github.com/NVIDIA/Morpheus/tree/branch-22.04/docs/source/developer_guide/guides
[Triton Inference Server Model Configuration]: https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md
[NVIDIA’s Cloud Native Core Stack]: https://github.com/NVIDIA/cloud-native-core
[NGC Registry CLI User Guide]: https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_4_1
