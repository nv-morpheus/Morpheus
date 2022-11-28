<!--
SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Prior to running the gnn fraud detection pipeline, additional requirements must be installed in to your conda environment. A supplemental requirements file has been provided in this example directory.

```bash
mamba env update -n ${CONDA_DEFAULT_ENV} -f examples/gnn_fraud_detection_pipeline/requirements.yml
```

## Running

Use Morpheus to run the GNN fraud detection Pipeline with the transaction data. A pipeline has been configured in `run.py` with several command line options:

```bash
python run.py --help
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers

  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model
  --input_file PATH               Input filepath  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --training_file PATH            Training data file [required]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model
  --model-xgb-file PATH           The name of the XGB model that is deployed
  --model-hinsage-file PATH       The name of the trained HinSAGE model file path

  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided at `<MORPHEUS_ROOT>/models/dataset`, run the following:

```bash

python run.py
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=validation.csv, iterative=None, file_type=auto, repeat=1, filter_null=False, cudf_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <fraud-graph-construction-2; FraudGraphConstructionStage(training_file=training.csv)>
  └─ morpheus.MultiMessage -> stages.FraudGraphMultiMessage
Added stage: <monitor-3; MonitorStage(description=Graph construction rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ stages.FraudGraphMultiMessage -> stages.FraudGraphMultiMessage
Added stage: <gnn-fraud-sage-4; GraphSAGEStage(model_hinsage_file=model/hinsage-model.pt, batch_size=5, sample_size=[2, 32], record_id=index, target_node=transaction)>
  └─ stages.FraudGraphMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <monitor-5; MonitorStage(description=Inference rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ stages.GraphSAGEMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <gnn-fraud-classification-6; ClassificationStage(model_xgb_file=model/xgb-model.pt)>
  └─ stages.GraphSAGEMultiMessage -> morpheus.MultiMessage
Added stage: <monitor-7; MonitorStage(description=Add classification rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.MultiMessage -> morpheus.MultiMessage
Added stage: <serialize-8; SerializeStage(include=None, exclude=['^ID$', '^_ts_'], output_type=pandas)>
  └─ morpheus.MultiMessage -> pandas.DataFrame
Added stage: <monitor-9; MonitorStage(description=Serialize rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ pandas.DataFrame -> pandas.DataFrame
Added stage: <to-file-10; WriteToFileStage(filename=result.csv, overwrite=True, file_type=auto)>
  └─ pandas.DataFrame -> pandas.DataFrame
====Building Pipeline Complete!====
====Pipeline Started====
Graph construction rate[Complete]: 265messages [00:00, 1590.22messages/s]
Inference rate[Complete]: 265messages [00:01, 150.23messages/s]
Add classification rate[Complete]: 265messages [00:01, 147.11messages/s]
Serialize rate[Complete]: 265messages [00:01, 142.31messages/s]
```

### CLI Example
The above example is illustrative of using the Python API to build a custom Morpheus Pipeline. Alternately the Morpheus command line could have been used to accomplish the same goal. To do this we must ensure that the `examples` directory is available in the `PYTHONPATH` and each of the custom stages are registered as plugins.
Note: Since the `gnn_fraud_detection_pipeline` module is visible to Python we can specify the plugins by their module name rather than the more verbose file path.

From the root of the Morpheus repo run:
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
	gnn-fraud-sage --model_hinsage_file examples/gnn_fraud_detection_pipeline/model/hinsage-model.pt \
	monitor --description "Inference rate" \
	gnn-fraud-classification --model_xgb_file examples/gnn_fraud_detection_pipeline/model/xgb-model.pt \
	monitor --description "Add classification rate" \
	serialize \
	to-file --filename "output.csv" --overwrite
```
