#!/bin/bash
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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh

update_conda_env

log_toolchain

cd ${MORPHEUS_ROOT}

fetch_base_branch

git submodule update --init --recursive

rapids-logger "Configuring cmake for Morpheus"
CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF"
export CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_INPLACE_BUILD=ON"
if [[ "${LOCAL_CI}" == "" ]]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DCCACHE_PROGRAM_PATH=$(which sccache)"
fi

cmake -B build -G Ninja ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build build --parallel ${PARALLEL_LEVEL}

if [[ "${LOCAL_CI}" == "" ]]; then
    rapids-logger "sccache usage for source build:"
    sccache --show-stats
fi

rapids-logger "Installing Morpheus"
pip install ./

# Setting this prevents loading of cudf since we don't have a GPU
export MORPHEUS_IN_SPHINX_BUILD=1

rapids-logger "Checking copyright headers"
python ${MORPHEUS_ROOT}/ci/scripts/copyright.py --verify-apache-v2 --git-diff-commits ${CHANGE_TARGET} ${GIT_COMMIT}

rapids-logger "Running Python style checks"
${MORPHEUS_ROOT}/ci/scripts/python_checks.sh

rapids-logger "Checking versions"
${MORPHEUS_ROOT}/ci/scripts/version_checks.sh

rapids-logger "Runing C++ style checks"
${MORPHEUS_ROOT}/ci/scripts/cpp_checks.sh
