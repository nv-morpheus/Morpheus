#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e

echo "Env Setup"
source /opt/conda/etc/profile.d/conda.sh
export MORPHEUS_ROOT=$(pwd)
env | sort

conda create -q -y -n morpheus python=${PYTHON_VER}
conda activate morpheus
conda config --set ssl_verify false
conda config --add pkgs_dirs /opt/conda/pkgs
conda config --env --add channels conda-forge
conda install -q -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" python=${PYTHON_VER}
mamba install -q -y -c gpuci gpuci-tools

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Building cuDF"
CONDA_BLD_DIR=/opt/conda/conda-bld
mkdir -p ${CONDA_BLD_DIR}
CONDA_ARGS="--output-folder ${CONDA_BLD_DIR} --skip-existing --no-test" ${MORPHEUS_ROOT}/ci/conda/recipes/run_conda_build.sh libcudf cudf

gpuci_logger "Installing cuDF"
mamba install -q -y -c file://${CONDA_BLD_DIR} -c nvidia -c rapidsai -c conda-forge libcudf cudf

gpuci_logger "Installing dependencies"
conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
mamba env update -q -n morpheus -f ./docker/conda/environments/cuda${CUDA_VER}_dev.yml

gpuci_logger "Check versions (cmake edition)"
python3 --version
gcc --version
g++ --version
cmake --version
ninja --version

gpuci_logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja -DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMORPHEUS_BUILD_BENCHMARKS=ON -DMORPHEUS_BUILD_EXAMPLES=ON -DMORPHEUS_BUILD_TESTS=ON -DMORPHEUS_USE_CONDA=ON .

gpuci_logger "Building Morpheus"
cmake --build build -j

gpuci_logger "Installing Morpheus"
pip install -e ${MORPHEUS_ROOT}
