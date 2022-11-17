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

# ============== LEGEND =================================================== #
#                         [conda_bld_morpheus]
#                               /
# [base] -> [conda_env] -> [base_extended] -> [runtime]
#                               \
#                         [conda_env_dev] -> [development] -> [development_pydbg]
#
#  base: Contains all system packages common across all environments
#  conda_env: Create morpheus conda environment and set it as the new base
#  base_extended: Optional environmental elements, libraries, packages, etc.
#  conda_env_dev: Full development environment with all necessary conda
#    packages.
#  conda_bld_morpheus: Utility stage used to build and cache the Morpheus
#    conda package for runtime environments. (should not be extended)
#  runtime: Full runtime environment with Morpheus installed as a conda
#    package
#  development: Full development environment able to build against a Morpheus
#    source directory.
#  development_pydbg: Development + debug build of cpython and various GDB
#    debugging macros.
# ========================================================================= #

# Args used in FROM commands must come first
ARG FROM_IMAGE="nvidia/cuda"
ARG CUDA_VER=11.7.0
ARG LINUX_DISTRO=ubuntu
ARG LINUX_VER=20.04

# ============ Stage: base ============
# Configure the base conda environment
FROM ${FROM_IMAGE}:${CUDA_VER}-devel-${LINUX_DISTRO}${LINUX_VER} AS base

# Required arguments
ARG BUILD_DOCA_DEPS="FALSE"
ARG CONDA_CHANNEL=rapidsai
ARG DOCA_ARTIFACTS_HOST=""
ARG DOCA_REPO_HOST=""
ARG IMAGE_TYPE=base
ARG PYTHON_VER=3.8
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=22.08

# Temp option to set conda channel_alias property during build
ARG CONDA_CHANNEL_ALIAS

# The TensorRT Version must be the full version (i.e. 7.2.2.3) and match Triton EXACTLY
ARG TENSORRT_VERSION=8.2.1.3

# Capture argument used for FROM
ARG CUDA_VER

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies to build
RUN apt-get update &&\
    apt-get upgrade -y &&\
    curl -sL https://deb.nodesource.com/setup_16.x | bash - &&\
    apt-get install --no-install-recommends -y \
        build-essential pkg-config curl unzip tar zip openssh-client bc jq git-lfs wget \
    && rm -rf /var/lib/apt/lists/*

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# All code will be under /workspace
WORKDIR /workspace

COPY --from=condaforge/mambaforge:latest /opt/conda /opt/conda

ENV PATH="${PATH}:/opt/conda/bin"

# RUN export PATH=/opt/conda/bin:$PATH

# Install mamba to speed the solve up
RUN conda config --set ssl_verify false &&\
    conda config --add pkgs_dirs /opt/conda/pkgs &&\
    conda config --env --add channels conda-forge &&\
    # Install mamba, boa and git here. Conda build breaks with other git installs
    /opt/conda/bin/conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" "git >=2.35.3" "git-lfs"  \
      "python=${PYTHON_VER}" &&\
    source activate base &&\
    git lfs install
    # conda clean -afy

# ============ Stage: conda_env ============
# Create the conda environment and install all dependencies
FROM base as conda_env

ENV TINI_VERSION 0.19.0

# Create a base environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Create the environment and install as little dependencies as possible
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba create -n morpheus -c conda-forge  \
      python=${PYTHON_VER} tini=${TINI_VERSION} && \
    # Clean and activate
    # conda clean -afy && \
    sed -i 's/conda activate base/conda activate morpheus/g' ~/.bashrc

# Set the permenant conda channes to use for morpheus
RUN source activate morpheus &&\
    conda config --env --add channels conda-forge &&\
    conda config --env --add channels nvidia &&\
    conda config --env --add channels rapidsai

# Add Tini
#ENV TINI_VERSION v0.19.0
#ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
#RUN chmod +x /tini

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
COPY docker/entrypoint.sh ./docker/
ENTRYPOINT [ "/opt/conda/envs/morpheus/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

# Reset the shell back to normal
SHELL ["/bin/bash", "-c"]

# ============ Stage: base_extended ============
# Base environment with any optional extensions. ex. DOCA libs
FROM conda_env as base_extended

# Add one or more optional dependencies to the base environment

ARG CONDA_CHANNEL_ALIAS
ARG BUILD_DOCA_DEPS

# Copy the source
COPY ./thirdparty thirdparty

# Required by DOCA runtime
# install doca dependencies from local deb packages
RUN if [[ ${BUILD_DOCA_DEPS} == @(TRUE|True|true|1) ]]; then \
        mkdir -p ./docker/deb && \
        wget --quiet -O ./docker/deb/doca-host-repo.deb \
          http://${DOCA_REPO_HOST}/doca-host-repo-ubuntu2004_1.5.0-0.1.1.1.5.0038.1.5.8.0.3.3.0_amd64.deb && \
        wget --quiet -O ./docker/deb/mlnx-dpdk.deb \
          https://${DOCA_ARTIFACTS_HOST}/mlnx-dpdk_20.11.0-6.1.2_amd64.deb && \
        wget --quiet -O ./docker/deb/mlnx-dpdk-dev.deb \
          https://${DOCA_ARTIFACTS_HOST}/mlnx-dpdk-dev_20.11.0-6.1.2_amd64.deb && \
        wget --quiet -O ./docker/deb/doca-libs.deb \
          https://${DOCA_ARTIFACTS_HOST}/doca-libs_1.5.0042-1_amd64.deb && \
        wget --quiet -O ./docker/deb/libdoca-libs-dev.deb \
          https://${DOCA_ARTIFACTS_HOST}/libdoca-libs-dev_1.5.0042-1_amd64.deb && \
        wget --quiet -O ./docker/deb/doca-apps.deb \
          https://${DOCA_ARTIFACTS_HOST}/doca-apps_1.5.0042-1_amd64.deb && \
        wget --quiet -O ./docker/deb/doca-apps-dev.deb \
          https://${DOCA_ARTIFACTS_HOST}/doca-apps-dev_1.5.0042-1_amd64.deb && \
        dpkg -i ./docker/deb/*.deb; exit 0 && \
        rm ./docker/deb/*.deb && \
        apt-get update && \
        apt-get install -y -f &&\
        rm -rf /var/lib/apt/lists/* ;\
    fi

# Build and install md, bsd, and gdrcopy libs
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    if [[ ${BUILD_DOCA_DEPS} == @(TRUE|True|true|1) ]]; then \
        source activate base && \
        mamba install automake cmake libtool -y && \
        cmake -DMORPHEUS_TP_BUILD_DOCA_DEPS=ON -DMORPHEUS_TP_INSTALL_DOCA_DEPS=ON \
          -B ./tp_build -S ./thirdparty/artifacts && \
        cmake --build ./tp_build && \
        mamba uninstall automake cmake libtool -y && \
        rm -rf ./tp_build ./thirdparty;\
    fi

# ============ Stage: conda_env_dev ============
# Create the development conda environment
FROM base_extended as conda_env_dev

ARG CONDA_CHANNEL_ALIAS

# Copy the development dependencies file
COPY docker/conda/environments/cuda${CUDA_VER}_dev.yml ./docker/conda/environments/
COPY docker/conda/environments/requirements.txt ./docker/conda/environments/

# Update the morpheus environment
RUN --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    # Temp add channel_alias to get around conda 404 errors
    conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"} &&\
    /opt/conda/bin/mamba env update -n morpheus --file docker/conda/environments/cuda${CUDA_VER}_dev.yml &&\
    # Remove channel_alias to use the normal channel in the container
    conda config --env --remove-key channel_alias &&\
    # Clean and activate
    conda clean -afy

# ============ Stage: conda_bld_morpheus ============
# Now build the morpheus conda package
FROM base_extended as conda_bld_morpheus

ARG CONDA_CHANNEL_ALIAS

# Copy the source
COPY . ./

RUN --mount=type=cache,id=workspace_cache,target=/workspace/.cache,sharing=locked \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate base &&\
    # Need to get around recent versions of git locking paths until they are deemed safe
    git config --global --add safe.directory "*" &&\
    MORPHEUS_ROOT=/workspace MORPHEUS_BUILD_PYTHON_STUBS=OFF CONDA_BLD_PATH=/opt/conda/conda-bld CONDA_ARGS="--no-test" \
      ./ci/conda/recipes/run_conda_build.sh morpheus

# sid_visualization is a submodule we need to init
RUN git submodule update --init --recursive

# ============ Stage: runtime ============
# Setup container for runtime environment
FROM base_extended as runtime

# Manually need to install some pip-only dependencies. Once these can get moved to conda, they can be removed.
COPY docker/conda/environments/requirements.txt ./docker/conda/environments/
COPY docker/conda/environments/cuda${CUDA_VER}_runtime.yml ./docker/conda/environments/

# Mount Morpheus conda package build in `conda_bld_morpheus`
RUN --mount=type=bind,from=conda_bld_morpheus,source=/opt/conda/conda-bld,target=/opt/conda/conda-bld \
    --mount=type=cache,id=conda_pkgs,target=/opt/conda/pkgs,sharing=locked \
    source activate morpheus &&\
    # Install morpheus
    CONDA_ALWAYS_YES=true /opt/conda/bin/mamba install -n morpheus  \
      -c local -c rapidsai -c nvidia -c nvidia/label/dev -c conda-forge \
      morpheus &&\
    # Install runtime dependencies that are pip-only
    /opt/conda/bin/mamba env update -n morpheus --file docker/conda/environments/cuda${CUDA_VER}_runtime.yml &&\
    # Install jupyter support (e.g., DFP)
    # TODO: this might not be the right spot to get these
    /opt/conda/bin/mamba install -n morpheus ipywidgets jupyterlab nb_conda_kernels &&\
    # Clean and activate
    conda clean -afy

# Only copy specific files/folders over that are necessary for runtime
COPY "./docker" "./docker"
COPY "./docs" "./docs"
COPY "./examples" "./examples"
COPY "./models" "./models"
COPY "./scripts" "./scripts"
COPY ["*.md", "LICENSE", "./"]

# remedy for CVE-2015-20107
RUN find / -name '*mailcap*.*py*' | xargs rm

# Use morpheus by default
CMD [ "morpheus" ]

# ============ Stage: development ============
# Install and configure development only packages
FROM conda_env_dev as development
# Copy the source
# COPY . ./

# Install node
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/include/node /usr/local/include/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
# Install yarn
COPY --from=node /opt/yarn-v*/bin/* /usr/local/bin/
COPY --from=node /opt/yarn-v*/lib/* /usr/local/lib/
# Copy entrypoint
COPY --from=node /usr/local/bin/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Install npm
RUN bash -c 'echo -e "\
    fund=false\n\
    audit=false\n\
    save-prefix=\n\
    --omit=optional\n\
    save-exact=true\n\
    package-lock=false\n\
    update-notifier=false\n\
    scripts-prepend-node-path=true\n\
    registry=https://registry.npmjs.org/\n\
    " | tee /root/.npmrc >/dev/null' \
     && ln -s /usr/local/bin/node /usr/local/bin/nodejs \
     && ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
     && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
     # Smoke tests
     && echo "node version: $(node --version)" \
     && echo " npm version: $(npm --version)" \
     && echo "yarn version: $(yarn --version)"

# ENV PATH="${PATH}:/usr/local/bin"

# Install camouflage needed for unittests to mock a triton server
RUN npm install -g camouflage-server

# Setup git to allow other users to access /workspace. Requires git 2.35.3 or
# greater. See https://marc.info/?l=git&m=164989570902912&w=2. Only enable for
# development
RUN git config --global --add safe.directory "*"

# ============ Stage: python_debug_bld ===========
# Configure and build cpython with debugging symbols
FROM development as development_pydbg

COPY ci/conda/recipes/python-dbg/ ./ci/conda/recipes/python-dbg
COPY ci/conda/recipes/run_conda_build.sh ./ci/conda/recipes/run_conda_build.sh
COPY ci/conda/recipes/python_dbg_install.sh ./ci/conda/recipes/python_dbg_install.sh

# Temporary until #68 goes in
ARG MORPHEUS_USER="root"

# Build and install debug cpython
RUN source activate morpheus \
    && MORPHEUS_ROOT=/workspace \
       CONDA_BLD_PATH=/opt/conda/conda-bld \
       CONDA_ARGS="--no-test" \
       ./ci/conda/recipes/run_conda_build.sh pydebug \
    && ./ci/conda/recipes/python_dbg_install.sh \
       -s $( ls /opt/conda/conda-bld/src_cache/Python-${PYTHON_VER}*.tar.xz ) \
       -c $( ls /opt/conda/conda-bld/linux-64/python-${PYTHON_VER}*.tar.bz2 ) \
       -i ./ci/conda/recipes/python-dbg/source \
       -g $(eval echo "~${MORPHEUS_USER}")/.gdbinit
