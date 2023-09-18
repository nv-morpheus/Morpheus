# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_PKG_VER=11-8
ARG CUDA_SHORT_VER=11.8
ARG CUDA_VER=11.8.0
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PROJ_NAME=morpheus
ARG PYTHON_VER=3.10

# Configure the base docker img
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

ARG PROJ_NAME
ARG CUDA_SHORT_VER

SHELL ["/bin/bash",  "-c"]

# Create conda environment
COPY ./docker/conda/environments/* /tmp/conda/

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n ${PROJ_NAME} -q --file /tmp/conda/cuda${CUDA_SHORT_VER}_dev.yml && \
    /opt/conda/bin/mamba install -n morpheus -c conda-forge "conda-merge>=0.2" && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda


# ============ build ==================
FROM base as build

# Add any build only dependencies here.

# ============ test ==================
FROM base as test

# Add any test only dependencies here.

ARG PROJ_NAME
ARG CUDA_SHORT_VER

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        openjdk-11-jre-headless && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./docker/conda/environments/cuda${CUDA_SHORT_VER}_examples.yml /tmp/conda/cuda${CUDA_SHORT_VER}_examples.yml
COPY ./ci/scripts/download_kafka.py /tmp/scripts/download_kafka.py

# Install camouflage needed for unittests to mock a triton server
RUN source activate ${PROJ_NAME} && \
    npm install -g camouflage-server@0.15 && \
    npm cache clean --force

# Install Kafka
RUN source activate ${PROJ_NAME} && \
    python /tmp/scripts/download_kafka.py && \
    rm -rf /tmp/scripts
