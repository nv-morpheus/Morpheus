#!/bin/bash
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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


echo "=== Setting up GLiNER ONNX Model for Triton ==="

TRITON_MODEL_REPO="./triton_models"
SOURCE_DIR="gliner_bi_encoder" # From your conversion script
python script/convert_gliner_onnx.py
# Create model directory for the pure ONNX model
mkdir -p ${TRITON_MODEL_REPO}/gliner_bi_encoder/1

# Copy ONNX model
if [ -f "${SOURCE_DIR}/model.onnx" ]; then
    cp ${SOURCE_DIR}/model.onnx ${TRITON_MODEL_REPO}/gliner_bi_encoder/1/
    echo " Copied ONNX model to gliner_bi_encoder/1/"
else
    echo " ONNX model not found at ${SOURCE_DIR}/model.onnx"
    echo "Please run convert_biencoder.py first"
    exit 1
fi

# Copy the config file to the model directory
if [ -f "${TRITON_MODEL_REPO}/${SOURCE_DIR}/config.pbtxt" ]; then
    cp ${TRITON_MODEL_REPO}/config.pbtxt ${TRITON_MODEL_REPO}/${SOURCE_DIR}/config.pbtxt
else
    echo " config.pbtxt not found for gliner_bi_encoder model."
    exit 1
fi

echo "GLiNER Pure ONNX model setup complete!"
echo "You can now start Triton server."
