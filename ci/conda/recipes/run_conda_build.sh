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

NUMARGS=$#
ARGS=$*

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function get_version() {
   echo "$(git describe --tags | grep -o -E '^([^-]*?)')"
}

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)}

# Change this to switch between build/mambabuild/debug
export CONDA_COMMAND=${CONDA_COMMAND:-"mambabuild"}

# Get the path to the morpheus git folder
export MORPHEUS_ROOT=${MORPHEUS_ROOT:-$(git rev-parse --show-toplevel)}

# Set the tag for the neo commit to use
export NEO_GIT_TAG=${NEO_GIT_TAG:-"5b55e37c6320c1a5747311a1e29e7ebb049d12bc"}

export CUDA="$(conda list | grep cudatoolkit | egrep -o "[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+")"
export PYTHON_VER="$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")"
export CUDA=11.4.1
echo "CUDA        : ${CUDA}"
echo "PYTHON_VER  : ${PYTHON_VER}"
echo "NEO_GIT_TAG : ${NEO_GIT_TAG}"
echo ""

# Export variables for the cache
export MORPHEUS_CACHE_DIR=${MORPHEUS_CACHE_DIR:-"${MORPHEUS_ROOT}/.cache"}

# Ensure the build directory exists
export CONDA_BLD_DIR=${CONDA_BLD_DIR:-"${MORPHEUS_CACHE_DIR}/conda-build"}
mkdir -p ${CONDA_BLD_DIR}

# Where the conda packages are saved to outside of the conda environment
CONDA_BLD_OUTPUT=${CONDA_BLD_OUTPUT:-"${MORPHEUS_ROOT}/.conda-bld"}

# Export CCACHE variables
export CCACHE_DIR="${MORPHEUS_CACHE_DIR}/ccache"
export CCACHE_NOHASHDIR=1
export CMAKE_GENERATOR="Ninja"
export CMAKE_C_COMPILER_LAUNCHER="ccache"
export CMAKE_CXX_COMPILER_LAUNCHER="ccache"
export CMAKE_CUDA_COMPILER_LAUNCHER="ccache"

# Holds the arguments in an array to allow for complex json objects
CONDA_ARGS_ARRAY=()

# Some default args
CONDA_ARGS_ARRAY+=("--use-local")

if [[ "${CONDA_COMMAND}" == "mambabuild" || "${CONDA_COMMAND}" == "build" ]]; then
   # Remove the timestamp from the work folder to allow caching to work better
   CONDA_ARGS_ARRAY+=("--build-id-pat" "{n}-{v}")
fi

# Choose default variants
CONDA_ARGS_ARRAY+=("--variants" "{python: 3.8}")

# And default channels
CONDA_ARGS_ARRAY+=("-c" "rapidsai" "-c" "nvidia" "-c" "nvidia/label/dev" "-c" "conda-forge")

if hasArg click_completion; then
   echo "Running conda-build for click_completion..."
   set -x
   conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/click_completion
   set +x
fi

if hasArg libneo; then

   export NEO_ROOT="${MORPHEUS_CACHE_DIR}/src_cache/libneo"
   export NEO_CACHE_DIR=${MORPHEUS_CACHE_DIR}

   # First need to download the repo into the cache
   if [[ ! -d "${NEO_ROOT}" ]]; then
      git clone ${NEO_GIT_URL:?"Cannot build libneo. Must set NEO_GIT_URL to git repo location to allow checkout of neo repository"} ${NEO_ROOT}
   fi

   pushd ${NEO_ROOT}

   # Ensure we have the latest checkout
   git fetch
   git checkout ${NEO_GIT_TAG}

   if [[ "$(git branch --show-current | wc -l)" == "1" ]]; then
      git pull
   fi

   # Set GIT_VERSION to set the project version inside of meta.yaml
   export GIT_VERSION="$(get_version)"

   echo "Running conda-build for libneo..."
   set -x
   conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/libneo
   set +x

   unset GIT_DESCRIBE_TAG

   popd
fi

if hasArg libcudf; then
   echo "Running conda-build for libcudf..."
   set -x
   conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/libcudf
   set +x
fi

if hasArg cudf; then
   echo "Running conda-build for cudf..."
   set -x
   conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/cudf
   set +x
fi

if hasArg morpheus; then
   # Set GIT_VERSION to set the project version inside of meta.yaml
   # Do this after neo in case they are different
   export GIT_VERSION="$(get_version)"

   echo "Running conda-build for morpheus..."
   set -x
   conda ${CONDA_COMMAND} "${CONDA_ARGS_ARRAY[@]}" ${CONDA_ARGS} ci/conda/recipes/morpheus
   set +x
fi
