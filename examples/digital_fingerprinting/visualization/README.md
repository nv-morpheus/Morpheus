<!--
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# Digital Fingerprinting (DFP) Visualization Pipeline

We show here how to set up and run the DFP pipeline on Azure and Duo log data to generate input files for the DFP visualization UI.

## Environment Setup

Follow the instructions [here](https://github.com/nv-morpheus/Morpheus/blob/branch-22.09/CONTRIBUTING.md) to set up your development environment in either a Docker container or conda environment.


## Download DFP Example Data from S3

```
pip install s3fs
```

```
./examples/digital_fingerprinting/fetch_example_data.py all
```

Azure training data will be saved to `examples/data/dfp/azure-training-data`, inference data to `examples/data/dfp/azure-inference-data`.
Duo training data will be saved to `examples/data/dfp/duo-training-data`, inference data to `examples/data/dfp/duo-inference-data`.

## Running pipeline for DFP Visualization

The pipeline uses `dfp-viz-postproc` CLI command to perform post-processing on DFP inference output. The inference output is converted to input format expected by the DFP Visualization and saves to multiple files based on specified time period. Time period to group data by must be [one of pandas' offset strings](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases). The default period is one day (D). The output files will be named by appending period to prefix (e.g. `dfp-viz-2022-08-30.csv`). These are the available options for `dfp-viz-postproc`:

```
--period                   Time period to group and save data by. [default: `D`]
--overwrite                Overwrite output files if they exist. [default: True]
--output_dir               Directory to which the output files will be written. [default: current directory]
--prefix                   Prefix for output files. [default: dfp-viz-]
```

Run the following to generate input files for Azure DFP visualization:
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_azure.txt \
--userid_column_name=userPrincipalName \
--timestamp_column_name=createdDateTime \
--feature_scaler=standard \
--use_generic_model \
from-azure \
--input_glob=examples/data/dfp/azure-inference-data/*.json \
--max_files=200 \
train-ae \
--train_data_glob=examples/data/dfp/azure-training-data/*.json \
--source_stage_class=morpheus.stages.input.azure_source_stage.AzureSourceStage \
--seed=42 \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
dfp-viz-postproc \
--period=D \
--prefix=dfp-viz-azure-
```

Run the following to generate input files for Azure DFP visualization:
```
morpheus --log_level=DEBUG \
run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=False \
pipeline-ae \
--columns_file=morpheus/data/columns_ae_duo.txt \
--userid_column_name=username \
--timestamp_column_name=time \
--feature_scaler=standard \
--use_generic_model \
from-duo \
--input_glob=examples/data/dfp/duo-inference-data/*.json \
--max_files=200 \
monitor --description='Input rate' \
train-ae \
--train_data_glob=examples/data/dfp/duo-training-data/*.json \
--source_stage_class=morpheus.stages.input.duo_source_stage.DuoSourceStage \
--seed=42 \
inf-pytorch \
monitor --description='Inference rate' --unit inf \
dfp-viz-postproc \
--period=D \
--prefix=dfp-viz-duo-
```
