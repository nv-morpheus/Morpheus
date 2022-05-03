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
export MORPHEUS_ROOT=$(pwd)
source /opt/conda/etc/profile.d/conda.sh
env | sort

#apt-get update
#apt-get upgrade -y

#apt-get install --no-install-recommends -y build-essential pkg-config curl unzip tar zip openssh-client bc jq

conda create -n morpheus python=${PYTHON_VER}
conda activate morpheus
conda config --env --add channels conda-forge
conda install -y -n base -c conda-forge "mamba >=0.22" "boa >=0.10" python=${PYTHON_VER}
mamba install -y -c rapidsai-nightly gpuci-tools

gpuci_logger "Check versions"
python3 --version
$CC --version
$CXX --version
echo $(which cmake)
cmake --version
echo $(which ninja)

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Building cuDF"
/docker/build_conda_packages.sh libcudf cudf

gpuci_logger "Installing cuDF"
mamba install -c file:///${MORPHEUS_ROOT}/.conda-bld -c nvidia -c rapidsai -c conda-forge libcudf cudf

gpuci_logger "Installing dependencies"
mamba env update -n morpheus -f ./docker/conda/environments/cuda${CUDA_VER}_dev.yml

gpuci_logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja -DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMORPHEUS_BUILD_BENCHMARKS=ON -DMORPHEUS_BUILD_EXAMPLES=ON -DMORPHEUS_BUILD_TESTS=ON -DMORPHEUS_USE_CONDA=ON .

gpuci_logger "Building Morpheus"
cmake --build build -j

gpuci_logger "Installing Morpheus"
pip install -e ${MORPHEUS_ROOT}
