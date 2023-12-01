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

git submodule update --init --recursive

CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_WHEEL=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF"
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
if [[ "${LOCAL_CI}" == "" ]]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DCCACHE_PROGRAM_PATH=$(which sccache)"
fi

rapids-logger "Configuring cmake for Morpheus with ${CMAKE_FLAGS}"
cmake -B build -G Ninja ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build build --parallel ${PARALLEL_LEVEL}

if [[ "${LOCAL_CI}" == "" ]]; then
    rapids-logger "sccache usage for morpheus build:"
    sccache --show-stats
fi

rapids-logger "Archiving results"
tar cfj "${WORKSPACE_TMP}/wheel.tar.bz" build/dist

MORPHEUS_LIBS=($(find ${MORPHEUS_ROOT}/build/morpheus/_lib -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;) \
                $(find ${MORPHEUS_ROOT}/examples -name "*.so" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/morhpeus_libs.tar.bz" "${MORPHEUS_LIBS[@]}"

CPP_TESTS=($(find ${MORPHEUS_ROOT}/build/morpheus/_lib/tests -name "*.x" -exec realpath --relative-to ${MORPHEUS_ROOT} {} \;))
tar cfj "${WORKSPACE_TMP}/cpp_tests.tar.bz" "${CPP_TESTS[@]}"

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact "${WORKSPACE_TMP}/wheel.tar.bz"
upload_artifact "${WORKSPACE_TMP}/morhpeus_libs.tar.bz"
upload_artifact "${WORKSPACE_TMP}/cpp_tests.tar.bz"

rapids-logger "Success"
exit 0
