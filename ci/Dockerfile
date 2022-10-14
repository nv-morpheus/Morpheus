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

# Args used in FROM commands must come first
ARG FROM_IMAGE="rapidsai/ci"
ARG CUDA_VER=11.5.1
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04
ARG PYTHON_VER=3.8

# Configure the base docker img
FROM ${FROM_IMAGE}:cuda${CUDA_VER}-${LINUX_DISTRO}${LINUX_VER}-py${PYTHON_VER} AS base

SHELL ["/bin/bash",  "-c"]

# OS deps
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    apt-get install --no-install-recommends -y \
        clang-format-12 \
        clang-tidy-12 \
        git-lfs \
        libclang-12-dev \
        libcublas-dev-11-5 \
        libcufft-dev-11-5 \
        libcurand-dev-11-5 \
        libcusolver-dev-11-5 \
        libnuma1 \
        libnvidia-compute-495 \
        llvm-12-dev \
        openjdk-11-jdk \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 121 \
        --slave /usr/bin/clang-apply-replacements clang-apply-replacements /usr/bin/clang-apply-replacements-12 \
        --slave /usr/bin/clang-format clang-format /usr/bin/clang-format-12 \
        --slave /usr/bin/clang-format-diff clang-format-diff /usr/bin/clang-format-diff-12 \
        --slave /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-12 \
        --slave /usr/bin/clang-tidy-diff.py clang-tidy-diff.py /usr/bin/clang-tidy-diff-12.py \
        --slave /usr/bin/clang++ clang++ /usr/bin/clang++-12 \
        --slave /usr/bin/git-clang-format git-clang-format /usr/bin/git-clang-format-12

# Create conda environment
COPY ./docker/conda/environments/cuda11.5_dev.yml /tmp
COPY ./docker/conda/environments/requirements.txt /tmp

RUN CONDA_ALWAYS_YES=true /opt/conda/bin/mamba env create -n morpheus -q --file /tmp/cuda11.5_dev.yml && \
    sed -i "s/conda activate base/conda activate morpheus/g" ~/.bashrc && \
    echo -e '#!/bin/sh\n\nexport CUDA_PATH=/usr/local/cuda\nexport LD_LIBRARY_PATH=/usr/local/cuda-11.5/compat:${LD_LIBRARY_PATH}' > /opt/conda/envs/morpheus/etc/conda/activate.d/env_vars.sh && \
    conda clean -afy && \
    rm /tmp/cuda11.5_dev.yml /tmp/requirements.txt

# Install camouflage needed for unittests to mock a triton server
RUN npm install -g camouflage-server && \
    npm cache clean --force

# Install pytest-kafka
# Installing pytest-kafka from source instead of conda/pip as the setup.py includes helper methods for downloading Kafka
# https://gitlab.com/karolinepauls/pytest-kafka/-/issues/9
RUN git clone https://gitlab.com/karolinepauls/pytest-kafka.git /opt/pytest-kafka && \
    cd /opt/pytest-kafka && \
    source activate morpheus && \
    python setup.py develop && \
    cd -
