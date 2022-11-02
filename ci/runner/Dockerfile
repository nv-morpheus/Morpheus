# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Note this is the Dockerfile for the CI runner not for morpheus itself
# Build from the root of the morpheus repo with:
# DOCKER_BUILDKIT=1 docker build --network=host --target test -t morpheus/ci:test-latest -f ci/runner/Dockerfile .
# DOCKER_BUILDKIT=1 docker build --network=host --target driver -t morpheus/ci:driver-latest -f ci/runner/Dockerfile .

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_VER=11.5.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8

# Configure the base docker img
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

ARG PROJ_NAME=morpheus
ARG CUDA_SHORT_VER=11.5
ARG CUDA_PKG_VER=11-5
ARG CLANG_VER=12

SHELL ["/bin/bash",  "-c"]

# OS deps
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        libnuma1 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY ./docker/conda/environments/* /tmp/conda

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n ${PROJ_NAME} -q --file /tmp/conda/cuda${CUDA_SHORT_VER}_dev.yml && \
    sed -i "s/conda activate base/conda activate ${PROJ_NAME}/g" ~/.bashrc && \
    conda clean -afy && \
    rm -rf /tmp/conda


# ============ driver ==================
FROM base as driver

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
    libcublas-dev-${CUDA_PKG_VER} \
    libcufft-dev-${CUDA_PKG_VER} \
    libcurand-dev-${CUDA_PKG_VER} \
    libcusolver-dev-${CUDA_PKG_VER} \
    libnvidia-compute-495 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


# ============ test ==================
FROM base as test

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt install --no-install-recommends -y \
        nodejs \
        npm \
        openjdk-11-jdk && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install camouflage needed for unittests to mock a triton server
RUN npm install -g camouflage-server@0.9 && \
    npm cache clean --force

# Install pytest-kafka
# Installing pytest-kafka from source instead of conda/pip as the setup.py includes helper methods for downloading Kafka
# https://gitlab.com/karolinepauls/pytest-kafka/-/issues/9
RUN git clone https://gitlab.com/karolinepauls/pytest-kafka.git /opt/pytest-kafka && \
    cd /opt/pytest-kafka && \
    source activate ${PROJ_NAME} && \
    python setup.py develop
