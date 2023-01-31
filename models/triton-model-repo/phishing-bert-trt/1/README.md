<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Generating TRT Models from ONNX

This model in the `triton-model-repo` directory is intentionally missing the model file. This is due to the fact that TensorRT maximizes performance of models for a *particular machine*. Any pre-compiled TensorRT engine file at best would have poor performance and most likely would not even load on other machines.

Therefore, it is best to compile a TensorRT engine file for on each machine that it will be run on. To facilitate this, Morpheus contains a utility to input an ONNX file and export the TensorRT engine file. To generate the necessary TensorRT engine file for this model, run the following from the same directory as this README:

```bash
morpheus tools onnx-to-trt --input_model ../../phishing-bert-onnx/1/model.onnx --output_model ./model.plan --batches 1 8 --batches 1 16 --batches 1 32 --seq_length 128 --max_workspace_size 16000
```

Note: If you get an out-of-memory error, reduce the `--max_workspace_size` argument until it will successfully run.
