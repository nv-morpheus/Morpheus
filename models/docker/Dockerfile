# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE="nvcr.io/nvidia/tritonserver"
ARG FROM_IMAGE_TAG="24.09-py3"
ARG MORPHEUS_ROOT_HOST=.
FROM --platform=$TARGETPLATFORM ${FROM_IMAGE}:${FROM_IMAGE_TAG} AS base

ARG MORPHEUS_ROOT_HOST

WORKDIR /

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt upgrade -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Append our third party notice to the entrypoint message
COPY "${MORPHEUS_ROOT_HOST}/thirdparty/models-container-thirdparty-oss.txt" "/opt/nvidia/entrypoint.d/80-morpheus-thirdparty-oss.txt"

# Copy the model repository
COPY "${MORPHEUS_ROOT_HOST}/models" "./models"

# Copy the example models that live in the examples dir
COPY "${MORPHEUS_ROOT_HOST}/examples/abp_pcap_detection/abp-pcap-xgb" "./models/triton-model-repo/abp-pcap-xgb"
