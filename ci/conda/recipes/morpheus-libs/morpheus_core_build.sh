# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# It is assumed that this script is executed from the root of the repo directory by conda-build
# (https://conda-forge.org/docs/maintainer/knowledge_base.html#using-cmake)

# This will store all of the cmake args. Make sure to prepend args to allow
# incoming values to overwrite them

source $RECIPE_DIR/cmake_common.sh

CMAKE_ARGS=${CMAKE_ARGS:-""}

export CCACHE_BASEDIR=$(realpath ${SRC_DIR}/..)
export USE_SCCACHE=${USE_SCCACHE:-""}

if [[ -n "${MORPHEUS_CACHE_DIR}" ]]; then
   # Set the cache variable, then set the Staging prefix to allow for host searching
   CMAKE_ARGS="-DMORPHEUS_CACHE_DIR=${MORPHEUS_CACHE_DIR} ${CMAKE_ARGS}"

   # Double check that the cache dir has been created
   mkdir -p ${MORPHEUS_CACHE_DIR}
fi

# Enable core. Core is enabled by default and this is to just highlight that it is on
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_CORE=ON ${CMAKE_ARGS}"

# Disable dfp, llm and doca
CMAKE_ARGS="-DMORPHEUS_SUPPORT_DOCA=OFF ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_DFP=OFF ${CMAKE_ARGS}"
CMAKE_ARGS="-DMORPHEUS_BUILD_MORPHEUS_LLM=OFF ${CMAKE_ARGS}"

if [[ "${USE_SCCACHE}" == "1" ]]; then
   CMAKE_ARGS="-DCCACHE_PROGRAM_PATH=$(which sccache) ${CMAKE_ARGS}"
fi

echo "CC          : ${CC}"
echo "CXX         : ${CXX}"
echo "CUDAHOSTCXX : ${CUDAHOSTCXX}"
echo "CUDA        : ${CUDA}"
echo "CMAKE_ARGS  : ${CMAKE_ARGS}"

echo "========Begin Env========"
env
echo "========End Env========"

BUILD_DIR="build-conda-core"

# remove the old build directory
if [[ -d "./${BUILD_DIR}" ]]; then
   echo "Deleting old build dir at ./${BUILD_DIR}"
   rm -rf "./${BUILD_DIR}/"
fi

# Run configure
cmake -B ${BUILD_DIR} \
   ${CMAKE_ARGS} \
   --log-level=verbose \
   .

# Build the components
cmake --build ${BUILD_DIR} -j${PARALLEL_LEVEL:-$(nproc)} --target install

# Install just the morpheus core python wheel components
${PYTHON} -m pip install -vv ${BUILD_DIR}/python/morpheus/dist/*.whl
