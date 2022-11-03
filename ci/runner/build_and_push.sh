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

DOCKER_TARGET=${DOCKER_TARGET:-"driver" "test"}
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
DOCKER_REGISTRY_SERVER=${DOCKER_REGISTRY_SERVER:-"nvcr.io"}
DOCKER_REGISTRY_PATH=${DOCKER_REGISTRY_PATH:-"/ea-nvidia-morpheus/morpheus"}
DOCKER_TAG_PREFIX=${DOCKER_TAG_PREFIX:-"morpheus-ci"}
DOCKER_TAG_POSTFIX=${DOCKER_TAG_POSTFIX:-"$(date +'%y%m%d')"}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

SKIP_BUILD=${SKIP_BUILD:-""}
SKIP_PUSH=${SKIP_PUSH:-""}

set -e

function get_image_full_name() {
   echo "${DOCKER_REGISTRY_SERVER}${DOCKER_REGISTRY_PATH}:${DOCKER_TAG_PREFIX}-${build_target}-${DOCKER_TAG_POSTFIX}"
}

if [[ "${SKIP_BUILD}" == "" ]]; then
    for build_target in ${DOCKER_TARGET[@]}; do
        FULL_NAME=$(get_image_full_name)
        echo "Building target \"${build_target}\" as ${FULL_NAME}";
        docker build --network=host ${DOCKER_EXTRA_ARGS} --target ${build_target} -t ${FULL_NAME} -f ci/runner/Dockerfile .
    done
fi

if [[ "${SKIP_PUSH}" == "" ]]; then
    for build_target in ${DOCKER_TARGET[@]}; do
        FULL_NAME=$(get_image_full_name)
        echo "Pushing ${FULL_NAME}";
        docker push ${FULL_NAME}
    done
fi
