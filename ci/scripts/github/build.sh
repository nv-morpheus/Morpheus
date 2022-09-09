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

source ${WORKSPACE}/ci/scripts/github/common.sh

gpuci_logger "Creating conda env"
rm -rf ${MORPHEUS_ROOT}/.cache/ ${MORPHEUS_ROOT}/build/
conda config --add pkgs_dirs /opt/conda/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
mamba env create -q -n morpheus -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_dev.yml
conda activate morpheus

gpuci_logger "Installing CI dependencies"
mamba env update -q -n morpheus -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_ci.yml
conda deactivate && conda activate morpheus

gpuci_logger "Final Conda Environment"
show_conda_info

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version
cmake --version
ninja --version

gpuci_logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DCCACHE_PROGRAM_PATH=$(which sccache) .

gpuci_logger "Building Morpheus"
cmake --build build -j --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for morpheus build:"
sccache --show-stats

gpuci_logger "Installing Morpheus"
cmake -DCOMPONENT=Wheel -P ${MORPHEUS_ROOT}/build/cmake_install.cmake

gpuci_logger "Archiving results"
mamba pack --quiet --force --ignore-missing-files --n-threads ${PARALLEL_LEVEL} -n morpheus -o ${WORKSPACE_TMP}/conda_env.tar.gz
tar cfj "${WORKSPACE_TMP}/wheel.tar.bz" build/wheel

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/conda_env.tar.gz" "${ARTIFACT_URL}/conda_env.tar.gz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/wheel.tar.bz" "${ARTIFACT_URL}/wheel.tar.bz"

gpuci_logger "Success"
exit 0
