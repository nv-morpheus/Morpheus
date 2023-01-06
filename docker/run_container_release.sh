#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Color variables
b="\033[0;36m"
g="\033[0;32m"
r="\033[0;31m"
e="\033[0;90m"
y="\033[0;33m"
x="\033[0m"

DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"nvcr.io/nvidia/morpheus/morpheus"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"$(git describe --tags --abbrev=0)-runtime"}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

DOCKER_ARGS="--runtime=nvidia --env WORKSPACE_VOLUME=${PWD} -v $PWD/models:/workspace/models --net=host --gpus=all --cap-add=sys_nice ${DOCKER_EXTRA_ARGS}"

if [[ -z "${SSH_AUTH_SOCK}" ]]; then
   echo -e "${y}No ssh-agent auth socket found. Dependencies in private git repos may fail during build.${x}"
else
   echo -e "${b}Setting up ssh-agent auth socket${x}"
   DOCKER_ARGS="${DOCKER_ARGS} -v $(readlink -f $SSH_AUTH_SOCK):/ssh-agent:ro -e SSH_AUTH_SOCK=/ssh-agent"
fi

echo -e "${g}Launching ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${x}"

docker run --rm -ti ${DOCKER_ARGS} ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} "${@:-bash}"
