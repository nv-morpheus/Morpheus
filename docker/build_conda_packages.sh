#!/bin/bash
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

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Need to build the development docker container to setup git safe.directory
export DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"morpheus"}
export DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"conda_build"}
export DOCKER_TARGET=${DOCKER_TARGET:-"development"}

MORPHEUS_ROOT=${MORPHEUS_ROOT:-$(git rev-parse --show-toplevel)}

echo "Building container"
# Call the build script to get a container ready to build conda packages
${SCRIPT_DIR}/build_container_dev.sh

# Now run the container with the volume mount to build the packages
export DOCKER_EXTRA_ARGS=""

echo "Running conda build"

CONDA_ARGS="--output-folder=/workspace/.conda-bld"
export DOCKER_EXTRA_ARGS="--env CONDA_ARGS=${CONDA_ARGS} --env NEO_GIT_URL=${NEO_GIT_URL:?"Cannot build libneo. Must set NEO_GIT_URL to git repo location to allow checkout of neo repository"}"

# Run with an output folder that is mounted and skip existing to avoid repeated builds
${SCRIPT_DIR}/run_container_dev.sh ./ci/conda/recipes/run_conda_build.sh libneo libcudf cudf

echo "Conda packages have been built. Use the following to install into an environment:"
echo "    mamba install -c file://$(realpath ${MORPHEUS_ROOT}/.conda-bld) -c nvidia -c rapidsai -c conda-forge neo cudf"
