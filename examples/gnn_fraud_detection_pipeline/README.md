<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# GNN Fraud Detection Pipeline

## Requirements

Prior to running the GNN fraud detection pipeline, additional requirements must be installed in to your Conda environment. A supplemental requirements file has been provided in this example directory.

```bash
export CUDA_VER=11.8
mamba install -n base -c conda-forge conda-merge
conda run -n base --live-stream conda-merge docker/conda/environments/cuda${CUDA_VER}_dev.yml \
  examples/gnn_fraud_detection_pipeline/requirements.yml > .tmp/merged.yml \
  && mamba env update -n ${CONDA_DEFAULT_ENV} --file .tmp/merged.yml
```

## Running

##### Setup Env Variable
```bash
export MORPHEUS_ROOT=$(pwd)
```

Use Morpheus to run the GNN fraud detection Pipeline with the transaction data. A pipeline has been configured in `run.py` with several command line options:

```bash
cd ${MORPHEUS_ROOT}/examples/gnn_fraud_detection_pipeline
python run.py --help
```
```
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use.
                                  [x>=1]
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers.  [x>=1]
  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model.  [x>=1]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model.
                                  [x>=1]
  --input_file PATH               Input data filepath.  [required]
  --training_file PATH            Training data filepath.  [required]
  --model_dir PATH                Trained model directory path  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided at `$MORPHEUS_ROOT/models/dataset`, run the following:

```bash
cd ${MORPHEUS_ROOT}/examples/gnn_fraud_detection_pipeline
python run.py
```
```
====Registering Pipeline====
====Building Pipeline====
Graph construction rate: 0 messages [00:00, ? me====Building Pipeline Complete!====
Inference rate: 0 messages [00:00, ? messages/s]====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started==== 0 messages [00:00, ? messages/s]
====Building Segment: linear_segment_0====ges/s]
Added source: <from-file-0; FileSourceStage(filename=validation.csv, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=False)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage(ensure_sliceable_index=True)>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <fraud-graph-construction-2; FraudGraphConstructionStage(training_file=training.csv)>
  └─ morpheus.MultiMessage -> stages.FraudGraphMultiMessage
Added stage: <monitor-3; MonitorStage(description=Graph construction rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ stages.FraudGraphMultiMessage -> stages.FraudGraphMultiMessage
Added stage: <gnn-fraud-sage-4; GraphSAGEStage(model_dir=model, batch_size=100, record_id=index, target_node=transaction)>
  └─ stages.FraudGraphMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <monitor-5; MonitorStage(description=Inference rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ stages.GraphSAGEMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <gnn-fraud-classification-6; ClassificationStage(model_xgb_file=model/xgb.pt)>
  └─ stages.GraphSAGEMultiMessage -> morpheus.MultiMessage
Added stage: <monitor-7; MonitorStage(description=Add classification rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.MultiMessage -> morpheus.MultiMessage
Added stage: <serialize-8; SerializeStage(include=[], exclude=['^ID$', '^_ts_'], fixed_columns=True)>
  └─ morpheus.MultiMessage -> morpheus.MessageMeta
Added stage: <monitor-9; MonitorStage(description=Serialize rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <to-file-10; WriteToFileStage(filename=output.csv, overwrite=True, file_type=FileTypes.Auto, include_index_col=True, flush=False)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Segment Complete!====
Graph construction rate[Complete]: 265 messages [00:00, 1218.88 messages/s]
Inference rate[Complete]: 265 messages [00:01, 174.04 messages/s]
Add classification rate[Complete]: 265 messages [00:01, 170.69 messages/s]
Serialize rate[Complete]: 265 messages [00:01, 166.36 messages/s]
====Pipeline Complete====
```

### CLI Example
The above example is illustrative of using the Python API to build a custom Morpheus pipeline. Alternately, the Morpheus command line could have been used to accomplish the same goal. To do this we must ensure the `examples` directory is available in the `PYTHONPATH` and each of the custom stages are registered as plugins.

Note: Since the `gnn_fraud_detection_pipeline` module is visible to Python we can specify the plugins by their module name rather than the more verbose file path.

From the root of the Morpheus repo, run:
```bash
PYTHONPATH="examples" \
morpheus --log_level INFO \
	--plugin "gnn_fraud_detection_pipeline" \
	run --use_cpp False --pipeline_batch_size 1024 --model_max_batch_size 32 --edge_buffer_size 4 \
	pipeline-other --model_fea_length 70 --label=probs \
	from-file --filename examples/gnn_fraud_detection_pipeline/validation.csv --filter_null False \
	deserialize \
	fraud-graph-construction --training_file examples/gnn_fraud_detection_pipeline/training.csv \
	monitor --description "Graph construction rate" \
	gnn-fraud-sage --model_dir  examples/gnn_fraud_detection_pipeline/model/ \
	monitor --description "Inference rate" \
	monitor --description "Add classification rate" \
	serialize \
	to-file --filename "output.csv" --overwrite
```
