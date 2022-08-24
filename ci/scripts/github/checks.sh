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

source ${WORKSPACE}/ci/scripts/jenkins/common.sh
export IWYU_DIR="${WORKSPACE_TMP}/iwyu"

gpuci_logger "Creating conda env"
rm -rf ${MORPHEUS_ROOT}/.cache/ ${MORPHEUS_ROOT}/build/ ${IWYU_DIR}
conda config --add pkgs_dirs /opt/conda/pkgs
conda config --env --add channels conda-forge
conda config --env --set channel_alias ${CONDA_CHANNEL_ALIAS:-"https://conda.anaconda.org"}
mamba env create -q -n morpheus -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_dev.yml
conda activate morpheus

gpuci_logger "Installing CI dependencies"
mamba env update -q -f ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_ci.yml

show_conda_info

gpuci_logger "Installing IWYU"
git clone https://github.com/include-what-you-use/include-what-you-use.git ${IWYU_DIR}
pushd ${IWYU_DIR}
git checkout clang_12
cmake -G Ninja \
    -DCMAKE_PREFIX_PATH=$(llvm-config --cmakedir) \
    -DCMAKE_C_COMPILER=$(which clang) \
    -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
    .

cmake --build . --parallel ${PARALLEL_LEVEL} --target install

popd

gpuci_logger "Runing Python style checks"
${MORPHEUS_ROOT}/ci/scripts/python_checks.sh

gpuci_logger "Configuring cmake for Morpheus"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DCCACHE_PROGRAM_PATH=$(which sccache) .

gpuci_logger "Building targets that generate source code"
cmake --build build --target style_checks --parallel ${PARALLEL_LEVEL}

gpuci_logger "sccache usage for source build:"
sccache --show-stats

gpuci_logger "Runing C++ style checks"
${MORPHEUS_ROOT}/ci/scripts/cpp_checks.sh

gpuci_logger "Checking copyright headers"
python ${MORPHEUS_ROOT}/ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits ${CHANGE_TARGET} ${GIT_COMMIT}
