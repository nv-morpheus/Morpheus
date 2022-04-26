# syntax=docker/dockerfile:1.3

# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARG FROM_IMAGE="gpuci/miniforge-cuda"
ARG CUDA_VER=11.4
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:${CUDA_VER}-devel-${LINUX_DISTRO}${LINUX_VER} AS base

# Required arguments
ARG IMAGE_TYPE=base
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=21.10
ARG PYTHON_VER=3.8
ARG CONDA_CHANNEL=rapidsai

# The TensorRT Version must be the full version (i.e. 7.2.2.3) and match Triton EXACTLY
ARG TENSORRT_VERSION=8.2.1.3

# Capture argument used for FROM
ARG CUDA_VER

# Install dependencies to build vcpkg dependencies
RUN apt-get update &&\
    apt-get upgrade -y &&\
    curl -sL https://deb.nodesource.com/setup_12.x | bash - &&\
    apt-get install --no-install-recommends -y \
        build-essential pkg-config curl unzip tar zip openssh-client bc jq nodejs \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# All code will be under /workspace
WORKDIR /workspace

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    conda config --env --add channels conda-forge &&\
    /opt/conda/bin/conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" python=${PYTHON_VER}
    # conda clean -afy

# ============ Stage: conda_bld_deps ============
# Now build the conda dependency packages
FROM base as conda_bld_deps

# COPY the files necessary to build the cudf package. Dont copy more otherwise the cache will get invalided often
COPY ci/conda/recipes/cudf/ ./ci/conda/recipes/cudf/
COPY ci/conda/recipes/libcudf/ ./ci/conda/recipes/libcudf/
COPY ci/conda/recipes/run_conda_build.sh ./ci/conda/recipes/run_conda_build.sh

RUN --mount=type=ssh \
    --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate base &&\
    # Run with --no-test for now until we can build with builtkit and default nvidia runtime
    MORPHEUS_ROOT=/workspace CONDA_BLD_DIR=/opt/conda/conda-bld CONDA_ARGS="--no-test" ./ci/conda/recipes/run_conda_build.sh libcudf cudf

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM base as conda_env

# Create a base environment
RUN --mount=type=bind,from=conda_bld_deps,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n morpheus -c conda-forge python=${PYTHON_VER} &&\
    # Clean and activate
    # conda clean -afy && \
    sed -i 's/conda activate base/conda activate morpheus/g' ~/.bashrc

# Set the permenant conda channes to use for morpheus
RUN source activate morpheus &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# ============ Stage: conda_env_dev ============
# Create the development conda environment
FROM conda_env as conda_env_dev

# Copy the development dependencies file
COPY docker/conda/environments/requirements.txt ./docker/conda/environments/
COPY docker/conda/environments/cuda${CUDA_VER}_dev.yml ./docker/conda/environments/

# Update the morpheus environment
RUN --mount=type=bind,from=conda_bld_deps,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    /opt/conda/bin/mamba env update -n morpheus --file docker/conda/environments/cuda${CUDA_VER}_dev.yml &&\
    # Clean and activate
    conda clean -afy

# ============ Stage: conda_bld_morpheus ============
# Now build the morpheus conda package
FROM conda_bld_deps as conda_bld_morpheus

# Copy the source
COPY . ./

RUN --mount=type=ssh \
    --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate base &&\
    MORPHEUS_ROOT=/workspace CONDA_BLD_DIR=/opt/conda/conda-bld CONDA_ARGS="--no-test" ./ci/conda/recipes/run_conda_build.sh morpheus

# ============ Stage: runtime ============
# Setup container for runtime environment
FROM conda_env as runtime

ARG MORPHEUS_USER="root"

# Manually need to install some pip-only dependencies. Once these can get moved to conda, they can be removed.
COPY docker/scripts/create_user.sh ./docker/scripts
COPY docker/conda/environments/requirements.txt ./docker/conda/environments/
COPY docker/conda/environments/cuda${CUDA_VER}_runtime.yml ./docker/conda/environments/

RUN docker/scripts/create_user.sh ${MORPHEUS_USER}
USER ${MORPHEUS_USER}
# TODO: do we need to chown any files?

RUN --mount=type=bind,from=conda_bld_morpheus,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate morpheus &&\
    # Install runtime dependencies that are pip-only
    /opt/conda/bin/mamba env update -n morpheus --file docker/conda/environments/cuda${CUDA_VER}_runtime.yml &&\
    # Install morpheus
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus -c local -c rapidsai -c nvidia -c nvidia/label/dev -c conda-forge morpheus &&\
    # Clean and activate
    conda clean -afy

# Only copy specific files/folders over that are necessary for runtime
COPY "./data" "./data"
COPY "./docker" "./docker"
COPY "./docs" "./docs"
COPY "./examples" "./examples"
COPY "./models" "./models"
COPY "./scripts" "./scripts"
COPY ["*.md", "LICENSE", "./"]

# Use morpheus by default
CMD [ "morpheus" ]

# ============ Stage: development ============
# Install and configure development only packages
FROM conda_env_dev as development

# Copy the source
# COPY . ./

# Install camouflage needed for unittests to mock a triton server
RUN npm install -g camouflage-server

# Setup git to allow other users to access /workspace. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"
