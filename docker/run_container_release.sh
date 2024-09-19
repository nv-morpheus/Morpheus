#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

# Change to the script file to ensure we are in the correct repo (in case were in a submodule)
pushd ${SCRIPT_DIR} &> /dev/null

MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA:-OFF}
MORPHEUS_BUILD_MORPHEUS_LLM=${MORPHEUS_BUILD_MORPHEUS_LLM:-ON}
MORPHEUS_BUILD_MORPHEUS_DFP=${MORPHEUS_BUILD_MORPHEUS_DFP:-ON}

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/morpheus"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0)-runtime"}

# This variable is used for passing extra arguments to the docker run command. Do not use DOCKER_ARGS for this purpose.
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

popd &> /dev/null

DOCKER_ARGS="--runtime=nvidia --env WORKSPACE_VOLUME=${PWD} --net=host --gpus=all --cap-add=sys_nice ${DOCKER_EXTRA_ARGS}"

if [[ -n "${SSH_AUTH_SOCK}" ]]; then
   echo -e "${b}Setting up ssh-agent auth socket${x}"
   DOCKER_ARGS="${DOCKER_ARGS} -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent:ro -e SSH_AUTH_SOCK=/ssh-agent"
fi

# DPDK requires hugepage and privileged container
DOCA_EXTRA_ARGS=""
if [[ ${MORPHEUS_SUPPORT_DOCA} == @(TRUE|ON) ]]; then
   echo -e "${b}Enabling DOCA Support. Mounting /dev/hugepages and running in privileged mode${x}"

   DOCKER_ARGS="${DOCKER_ARGS} -v /dev/hugepages:/dev/hugepages --privileged"
fi


echo -e "${g}Launching ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${x}"

# Enable command logging to show what is being executed
set -x
docker run ${DOCA_EXTRA_ARGS} --rm -ti ${DOCKER_ARGS} ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} "${@:-bash}"
set +x
