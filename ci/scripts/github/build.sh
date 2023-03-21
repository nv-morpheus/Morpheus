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

rapids-logger "Check versions"
python3 --version
x86_64-conda-linux-gnu-cc --version
x86_64-conda-linux-gnu-c++ --version
cmake --version
ninja --version
sccache --version

rapids-logger "Configuring cmake for Morpheus"
git submodule update --init --recursive

cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} \
    -DCCACHE_PROGRAM_PATH=$(which sccache) \
    -DMORPHEUS_PYTHON_BUILD_WHEEL=ON \
    -DMORPHEUS_PYTHON_BUILD_STUBS=OFF \
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc  .

rapids-logger "Building Morpheus"
cmake --build build --parallel ${PARALLEL_LEVEL}

rapids-logger "sccache usage for morpheus build:"
sccache --show-stats

rapids-logger "Archiving results"
tar cfj "${WORKSPACE_TMP}/wheel.tar.bz" build/dist

MORPHEUS_LIBS=($(find ${MORPHEUS_ROOT}/build/morpheus/_lib -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/morhpeus_libs.tar.bz" "${MORPHEUS_LIBS[@]}"

CPP_TESTS=($(find ${MORPHEUS_ROOT}/build/morpheus/_lib/tests -name "*.x" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/cpp_tests.tar.bz" "${CPP_TESTS[@]}"

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/wheel.tar.bz" "${ARTIFACT_URL}/wheel.tar.bz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/morhpeus_libs.tar.bz" "${ARTIFACT_URL}/morhpeus_libs.tar.bz"
aws s3 cp --no-progress "${WORKSPACE_TMP}/cpp_tests.tar.bz" "${ARTIFACT_URL}/cpp_tests.tar.bz"

rapids-logger "Success"
exit 0
