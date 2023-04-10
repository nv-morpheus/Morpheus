<!--
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# "Starter" Digital Fingerprinting Pipeline

We show here how to set up and run the DFP pipeline for three log types: CloudTrail, Duo and Azure. Each of these log types uses a built-in source stage that handles that specific data format. New source stages can be added to allow the DFP pipeline to process different log types. All stages after the source stages are identical across all log types but can be configured differently via pipeline or stage configuration options.

## Environment Setup

Follow the instructions [here](https://github.com/nv-morpheus/Morpheus/blob/branch-23.07/docs/source/developer_guide/contributing.md) to set up your development environment in either a Docker container or conda environment.

## Morpheus CLI

DFP pipelines can be constructed and run using the Morpheus CLI command `morpheus run pipeline-ae ...`

Use `--help` to display information about the autoencoder pipeline command line options:

```
morpheus run pipeline-ae --help

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
  --columns_file FILE             [default: ./morpheus/data/columns_ae_cloudtrail.txt]
  --labels_file FILE              Specifies a file to read labels from in
                                  order to convert class IDs into labels. A
                                  label file is a simple text file where each
                                  line corresponds to a label. If unspecified,
                                  only a single output label is created for
                                  FIL
  --userid_column_name TEXT       Which column to use as the User ID.
                                  [default: userIdentityaccountId; required]
  --userid_filter TEXT            Specifying this value will filter all
                                  incoming data to only use rows with matching
                                  User IDs. Which column is used for the User
                                  ID is specified by `userid_column_name`
  --feature_scaler TEXT           Autoencoder feature scaler  [default:
                                  standard]
  --use_generic_model BOOLEAN     Whether to use a generic model when user does
                                  not have minimum number of training rows
                                  [default: False]
  --viz_file FILE                 Save a visualization of the pipeline at the
                                  specified location
  --help                          Show this message and exit.

Commands:
  add-class        Add detected classifications to each message
  add-scores       Add probability scores to each message
  buffer           (Deprecated) Buffer results
  delay            (Deprecated) Delay results for a certain duration
  filter           Filter message by a classification threshold
  from-azure       Source stage is used to load Azure Active Directory messages.
  from-cloudtrail  Load messages from a Cloudtrail directory
  from-duo         Source stage is used to load Duo Authentication messages.
  gen-viz          (Deprecated) Write out visualization data frames
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
The commands above correspond to the Morpheus stages that can be used to construct your DFP pipeline. Options are available to configure pipeline and stages.
The following table shows mapping between the main Morpheus CLI commands and underlying Morpheus Python stage classes:

| CLI Command    | Stage Class              | Python File              |
| ---------------| -------------------------| ---------------------------------------------------------
| from-azure     | AzureSourceStage         | morpheus/stages/input/azure_source_stage.py
| from-cloudtrail| CloudTrailSourceStage    | morpheus/stages/input/clout_trail_source_stage.py
| from-duo       | DuoSourceStage           | morpheus/stages/input/duo_source_stage.py
| train-ae       | TrainAEStage             | morpheus/stages/preprocess/train_ae_stage.py
| preprocess     | PreprocessAEStage        | morpheus/stages/preprocess/preprocess_ae_stage.py
| inf-pytorch    | AutoEncoderInferenceStage| morpheus/stages/inference/auto_encoder_inference_stage.py
| add-scores     | AddScoresStage           | morpheus/stages/postprocess/add_scores_stage.py
| serialize      | SerializeStage           | morpheus/stages/postprocess/serialize_stage.py
| to-file        | WriteToFileStage         | morpheus/stages/output/write_to_file_stage.py


## Morpheus DFP Stages

**Source stages** - These include `AzureSourceStage`, `CloudTrailSourceStage` and `DuoSourceStage`. They are responsible for reading log file(s) that match provided `--input_glob` (e.g. `/duo_logs/*.json`). Data is grouped by user so that each batch processed by the pipeline will only contain rows corresponding to a single user. Feature engineering also happens in this stage. All DFP source stages must extend `AutoencoderSourceStage` and implement the `files_to_dfs_per_user` abstract method. Feature columns can be managed by overriding the `derive_features` method. Otherwise, all columns from input data pass through to next stage.

**Preprocessing stages**

`TrainAEStage` can either train user models using data matching a provided `--train_data_glob` or load pre-trained models from file using `--pretrained_filename`. When using `--train_data_glob`, user models can be saved using the `--models_output_filename` option. The `--source_stage_class` must also be used with `--train_data_glob` so that the training stage knows how to read the training data. The autoencoder implementation used for user model training can be found [here](https://github.com/nv-morpheus/dfencoder). The following are the available CLI options for the `TrainAEStage` (train-ae):

| Option                | Description
| ----------------------| ---------------------------------------------------------
| pretrained_filename   | File path to pickled user models saved from previous training run using `--models_output_filename`.
| train_data_glob       | Glob path to training data.
| source_stage_class    | Source stage so that training stage knows how to read/parse training data.
| train_epochs          | Number of training epochs. Default is 25.
| min_train_rows        | Minimum number of training rows required to train user model. Default is 300.
| train_max_history     | Maximum number of training rows per user. Default is 1000.
| seed                  | When not None, ensure random number generators are seeded with `seed` to control reproducibility of user model.
| sort_glob             | If true the list of files matching `input_glob` will be processed in sorted order. Default is False.
| models_output_filename| Can be used with `--train_data_glob` to save trained user models to file using provided file path. Models can be loaded later using `--pretrained_filename`.

The `PreprocessAEStage` is responsible for creating a Morpheus message that contains everything needed by the inference stage. For DFP inference, this stage must pass a `MultiInferenceAEMessage` to the inference stage. Each message will correspond to a single user and include the input feature columns, the user's model and training data anomaly scores.

**Inference stage** - `AutoEncoderInferenceStage` calculates anomaly scores (i.e. reconstruction loss) and z-scores for each user input dataset.

**Postprocessing stage** - The DFP pipeline uses the `AddScoresStage` for postprocessing to add anomaly scores and zscores from previous inference stage with matching labels.

**Serialize stage** - `SerializeStage` is used to convert `MultiResponseMessage` from previous stage to a `MessageMeta` to make it suitable for output (i.e. write to file or Kafka).

**Write stage** - `WriteToFileStage` writes input data with inference results to an output file path.

## Download DFP Example Data from S3

```
pip install s3fs
```

```
./examples/digital_fingerprinting/fetch_example_data.py all
```

Azure training data will be saved to `examples/data/dfp/azure-training-data`, inference data to `examples/data/dfp/azure-inference-data`.
Duo training data will be saved to `examples/data/dfp/duo-training-data`, inference data to `examples/data/dfp/duo-inference-data`.

## CloudTrail DFP Pipeline

Run the following in your Morpheus container to start the CloudTrail DFP pipeline:

```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_cloudtrail.txt \
--userid_column_name=userIdentitysessionContextsessionIssueruserName \
--userid_filter=user123 \
--feature_scaler=standard \
from-cloudtrail \
--input_glob=models/datasets/validation-data/dfp-cloudtrail-*-input.csv \
--max_files=200 \
train-ae \
--train_data_glob=models/datasets/training-data/dfp-cloudtrail-*.csv \
--source_stage_class=morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage \
--seed=42 \
preprocess \
inf-pytorch \
add-scores \
serialize \
to-file --filename=./cloudtrail-dfp-detections.csv --overwrite
```

## Duo DFP Pipeline

The following pipeline trains user models from downloaded training data and saves user models to file. Pipeline then uses these models to run inference
on downloaded inference data. Inference results are written to `duo-detections.csv`.
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_duo.txt \
--userid_column_name=username \
--feature_scaler=standard \
from-duo \
--input_glob=examples/data/dfp/duo-inference-data/*.json \
--max_files=200 \
monitor --description='Input rate' \
train-ae \
--train_data_glob=examples/data/dfp/duo-training-data/*.json \
--source_stage_class=morpheus.stages.input.duo_source_stage.DuoSourceStage \
--seed=42 \
--models_output_filename=models/dfp-models/duo_ae_user_models.pkl \
preprocess \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
add-scores \
serialize \
to-file --filename=./duo-detections.csv --overwrite
```

The following example shows how we can load pre-trained user models from the file (`models/dfp-models/duo_ae_user_models.pkl`) we created in the previous example. Pipeline then uses these models to run inference on validation data in `models/datasets/validation-data/duo`. Inference results are written to `duo-detections.csv`.
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_duo.txt \
--userid_column_name=username \
--feature_scaler=standard \
from-duo \
--input_glob=examples/data/dfp/duo-inference-data/*.json \
--max_files=200 \
monitor --description='Input rate' \
train-ae \
--pretrained_filename=models/dfp-models/duo_ae_user_models.pkl \
preprocess \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
add-scores \
serialize \
to-file --filename=./duo-detections.csv --overwrite
```

## Azure DFP Pipeline

The following pipeline trains user models from downloaded training data and saves user models to file. Pipeline then uses these models to run inference
on downloaded inference data. Inference results are written to `azure-detections.csv`.
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_azure.txt \
--userid_column_name=userPrincipalName \
--feature_scaler=standard \
from-azure \
--input_glob=examples/data/dfp/azure-inference-data/*.json \
--max_files=200 \
train-ae \
--train_data_glob=examples/data/dfp/azure-training-data/*.json \
--source_stage_class=morpheus.stages.input.azure_source_stage.AzureSourceStage \
--seed=42 \
--models_output_filename=models/dfp-models/azure_ae_user_models.pkl \
preprocess \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
add-scores \
serialize \
to-file --filename=./azure-detections.csv --overwrite
```

The following example shows how we can load pre-trained user models from the file (`models/dfp-models/azure_ae_user_models.pkl`) we created in the previous example. Pipeline then uses these models to run inference on validation data in `models/datasets/validation-data/azure`. Inference results are written to `azure-detections.csv`.
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_azure.txt \
--userid_column_name=userPrincipalName \
--feature_scaler=standard \
from-azure \
--input_glob=examples/data/dfp/azure-inference-data/*.json \
--max_files=200 \
train-ae \
--pretrained_filename=models/dfp-models/azure_ae_user_models.pkl \
preprocess \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
add-scores \
serialize \
to-file --filename=./azure-detections.csv --overwrite
```


## Using Morpheus Python API

The DFP pipelines can also be constructed and run via the Morpheus Python API. An [example](./run_cloudtrail_dfp.py) is included for the Cloudtrail DFP pipeline. The following are some commands to
run the example.

Train user models from files in `models/datasets/training-data/dfp-cloudtrail-*.csv` and saves user models to file. Pipeline then uses these models to run inference on Cloudtrail validation data in `models/datasets/validation-data/dfp-cloudtrail-*-input.csv`. Inference results are written to `cloudtrail-dfp-results.csv`.
```
python ./examples/digital_fingerprinting/starter/run_cloudtrail_dfp.py \
    --columns_file=morpheus/data/columns_ae_cloudtrail.txt \
    --input_glob=models/datasets/validation-data/dfp-cloudtrail-*-input.csv \
    --train_data_glob=models/datasets/training-data/dfp-*.csv \
    --models_output_filename=models/dfp-models/cloudtrail_ae_user_models.pkl \
    --output_file ./cloudtrail-dfp-results.csv
```

Here we load pre-trained user models from the file (`models/dfp-models/cloudtrail_ae_user_models.pkl`) we created in the previous example. Pipeline then uses these models to run inference on validation data in `models/datasets/validation-data/dfp-cloudtrail-*-input.csv`. Inference results are written to `cloudtrail-dfp-results.csv`.
```
python ./examples/digital_fingerprinting/starter/run_cloudtrail_dfp.py \
    --columns_file=morpheus/data/columns_ae_cloudtrail.txt \
    --input_glob=models/datasets/validation-data/dfp-cloudtrail-*-input.csv \
    --pretrained_filename=models/dfp-models/cloudtrail_ae_user_models.pkl \
    --output_file=./cloudtrail-dfp-results.csv
```
