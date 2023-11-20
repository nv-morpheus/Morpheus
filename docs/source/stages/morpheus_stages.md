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

# Stages Documentation

Stages are the building blocks of Morpheus pipelines. Below is a list of the most commonly used stages. For a full list of stages, refer to the stages API {py:mod}`morpheus.stages`.

## Doca

- Doca Stage {py:class}`~morpheus.stages.doca.doca_source_stage.DocaSourceStage` A source stage used to receive raw packet data from a ConnectX-6 Dx NIC. This stage is not compiled by default refer to the [Doca Example](../../../examples/doca/README.md) for details on building this stage.

## General

- Linear Modules Stage {py:class}`~morpheus.stages.general.linear_modules_stage.LinearModulesStage` Loads an existing, registered, module and wraps it as a Morpheus stage. Refer to [Morpheus Modules](../developer_guide/guides.md#morpheus-modules) for details on modules.
- Monitor Stage {py:class}`~morpheus.stages.general.monitor_stage.MonitorStage` Display throughput numbers at a specific point in the pipeline.
- Multi Port Module Stage {py:class}`~morpheus.stages.general.multi_port_modules_stage.MultiPortModulesStage` Loads an existing, registered, multi-port module and wraps it as a multi-port Morpheus stage. Refer to [Morpheus Modules](../developer_guide/guides.md#morpheus-modules) for details on modules.
- Trigger Stage {py:class}`~morpheus.stages.general.trigger_stage.TriggerStage` Buffer data until the previous stage has completed, useful for testing performance of one stage at a time.

## Inference

- Auto Encoder Inference Stage {py:class}`~morpheus.stages.inference.auto_encoder_inference_stage.AutoEncoderInferenceStage` PyTorch inference stage used for Auto Encoder pipeline mode.
- PyTorch Inference Stage {py:class}`~morpheus.stages.inference.pytorch_inference_stage.PyTorchInferenceStage` PyTorch inference stage used for most pipeline modes with the exception of Auto Encoder.
- Triton Inference Stage {py:class}`~morpheus.stages.inference.triton_inference_stage.TritonInferenceStage`  Inference stage which utilizes a [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server).

## Input

- AppShield Source Stage {py:class}`~morpheus.stages.input.appshield_source_stage.AppShieldSourceStage`
- Azure Source Stage {py:class}`~morpheus.stages.input.azure_source_stage.AzureSourceStage`
- Cloud Trail Source Stage {py:class}`~morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage`
- Control Message File Source Stage {py:class}`~morpheus.stages.input.control_message_file_source_stage.ControlMessageFileSourceStage`
- Control Message Kafka Source Stage {py:class}`~morpheus.stages.input.control_message_kafka_source_stage.ControlMessageKafkaSourceStage`
- Databricks Delta Lake Source Stage {py:class}`~morpheus.stages.input.databricks_deltalake_source_stage.DataBricksDeltaLakeSourceStage`
- Duo Source Stage {py:class}`~morpheus.stages.input.duo_source_stage.DuoSourceStage`
- File Source Stage {py:class}`~morpheus.stages.input.file_source_stage.FileSourceStage`
- HTTP Client Source Stage {py:class}`~morpheus.stages.input.http_client_source_stage.HttpClientSourceStage`
- HTTP Server Source Stage {py:class}`~morpheus.stages.input.http_server_source_stage.HttpServerSourceStage`
- In Memory Source Stage {py:class}`~morpheus.stages.input.in_memory_source_stage.InMemorySourceStage`
- Kafka Source Stage {py:class}`~morpheus.stages.input.kafka_source_stage.KafkaSourceStage`
- RSS Source Stage {py:class}`~morpheus.stages.input.rss_source_stage.RSSSourceStage`

## Output
- HTTP Client Sink Stage {py:class}`~morpheus.stages.output.http_client_sink_stage.HttpClientSinkStage`
- HTTP Server Sink Stage {py:class}`~morpheus.stages.output.http_server_sink_stage.HttpServerSinkStage`
- In Memory Sink Stage {py:class}`~morpheus.stages.output.in_memory_sink_stage.InMemorySinkStage`
- Databricks Delta Lake Sink Stage {py:class}`~morpheus.stages.output.write_to_databricks_deltalake_stage.DataBricksDeltaLakeSinkStage`
- Write To Elastic Search Stage {py:class}`~morpheus.stages.output.write_to_elasticsearch_stage.WriteToElasticsearchStage`
- Write To File Stage {py:class}`~morpheus.stages.output.write_to_file_stage.WriteToFileStage`
- Write To Kafka Stage {py:class}`~morpheus.stages.output.write_to_kafka_stage.WriteToKafkaStage`
- Write To Vector DB Stage {py:class}`~morpheus.stages.output.write_to_vector_db.WriteToVectorDBStage`

## Post-process

- Add Classifications Stage {py:class}`~morpheus.stages.postprocess.add_classifications_stage.AddClassificationsStage`
- Add Scores Stage {py:class}`~morpheus.stages.postprocess.add_scores_stage.AddScoresStage`
- Filter Detections Stage {py:class}`~morpheus.stages.postprocess.filter_detections_stage.FilterDetectionsStage`
- Generate Viz Frames Stage {py:class}`~morpheus.stages.postprocess.generate_viz_frames_stage.GenerateVizFramesStage`
- ML Flow Drift Stage {py:class}`~morpheus.stages.postprocess.ml_flow_drift_stage.MLFlowDriftStage`
- Serialize Stage {py:class}`~morpheus.stages.postprocess.serialize_stage.SerializeStage`
- Timeseries Stage {py:class}`~morpheus.stages.postprocess.timeseries_stage.TimeSeriesStage`

## Pre-process

- Deserialize Stage {py:class}`~morpheus.stages.preprocess.deserialize_stage.DeserializeStage`
- Drop Null Stage {py:class}`~morpheus.stages.preprocess.drop_null_stage.DropNullStage`
- Preprocess AE Stage {py:class}`~morpheus.stages.preprocess.preprocess_ae_stage.PreprocessAEStage`
- Preprocess FIL Stage {py:class}`~morpheus.stages.preprocess.preprocess_fil_stage.PreprocessFILStage`
- Preprocess NLP Stage {py:class}`~morpheus.stages.preprocess.preprocess_nlp_stage.PreprocessNLPStage`
- Train AE Stage {py:class}`~morpheus.stages.preprocess.train_ae_stage.TrainAEStage`
