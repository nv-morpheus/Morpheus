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

# Digital Fingerprinting (DFP) in Morpheus

## Organization

The DFP example workflows in Morpheus are designed to scale up to company wide workloads and handle several different log types which resulted in a large number of moving parts to handle the various services and configuration options. To simplify things, the DFP workflow is provided as two separate examples: a simple, "starter" pipeline for new users and a complex, "production" pipeline for full scale deployments. While these two examples both peform the same general tasks, they do so in very different ways. The following is a breakdown of the differences between the two examples.

### The "Starter" Example

This example is designed to simplify the number of stages and components and provided a fully contained workflow in a single pipeline.

Key Differences:
 * A single pipeline which performs both training and inference
 * Requires no external services
 * Can be run from the Morpheus CLI


### The "Production" Example

This example is designed to show what a full scale, production ready, DFP deployment in Morpheus would look like. It contains all of the necessary components (such as a model store), to allow multiple Morpheus pipelines to communicate at a scale that can handle the workload of an entire company.

Key Differences:
 * Multiple pipelines are specialized to perform either training or inference
 * Requires setting up a model store to allow the training and inference pipelines to communicate
 * Organized into a docker-compose deployment for easy startup
 * Contains a Jupyter notebook service to ease development and debugging
 * Can be deployed to Kubernetes using provided Helm charts
 * Uses many customized stages to maximize performance.

## Getting Started

Guides for each of the two examples can be found in their respective directories: [The Starter Example](./starter/README.md) and [The Production Example](./production/README.md)
