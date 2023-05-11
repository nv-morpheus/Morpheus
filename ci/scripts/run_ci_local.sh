#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

case "$1" in
    "" )
        STAGES=("bash")
        ;;
    "all" )
        STAGES=("check" "build" "docs" "test")
        ;;
    "check" | "build" | "docs" | "test" | "bash" )
        STAGES=("$1")
        ;;
    * )
        echo "Error: Invalid argument \"$1\" provided. Expected values: \"all\", \"check\", \"build\", \"docs\", \"test\", or \"bash\""
        exit 1
        ;;
esac

MORPHEUS_ROOT=${MORPHEUS_ROOT:-$(git rev-parse --show-toplevel)}
GIT_URL=$(git remote get-url origin | sed -e 's|^git@github\.com:|https://github.com/|')
GIT_BRANCH=$(git branch --show-current)
GIT_COMMIT=$(git log -n 1 --pretty=format:%H)

LOCAL_CI_TMP=${LOCAL_CI_TMP:-${MORPHEUS_ROOT}/.tmp/local_ci_tm}
CONTAINER_VER=${CONTAINER_VER:-230510}
CUDA_VER=${CUDA_VER:-11.8}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

BUILD_CONTAINER="nvcr.io/ea-nvidia-morpheus/morpheus:morpheus-ci-build-${CONTAINER_VER}"
TEST_CONTAINER="nvcr.io/ea-nvidia-morpheus/morpheus:morpheus-ci-test-${CONTAINER_VER}"

ENV_LIST="--env LOCAL_CI_TMP=/ci_tmp"
ENV_LIST="${ENV_LIST} --env GIT_URL=${GIT_URL}"
ENV_LIST="${ENV_LIST} --env GIT_BRANCH=${GIT_BRANCH}"
ENV_LIST="${ENV_LIST} --env GIT_COMMIT=${GIT_COMMIT}"
ENV_LIST="${ENV_LIST} --env PARALLEL_LEVEL=$(nproc)"
ENV_LIST="${ENV_LIST} --env CUDA_VER=${CUDA_VER}"

mkdir -p ${LOCAL_CI_TMP}
cp ${MORPHEUS_ROOT}/ci/scripts/bootstrap_local_ci.sh ${LOCAL_CI_TMP}

for STAGE in "${STAGES[@]}"; do
    DOCKER_RUN_ARGS="--rm -ti --net=host -v "${LOCAL_CI_TMP}":/ci_tmp ${ENV_LIST} --env STAGE=${STAGE}"
    if [[ "${STAGE}" == "test" || "${USE_GPU}" == "1" ]]; then
        CONTAINER="${TEST_CONTAINER}"
        DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --runtime=nvidia --gpus all"
        if [[ "${STAGE}" == "test" ]]; then
            DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --env MERGE_EXAMPLES_YAML=1"
        fi
    else
        CONTAINER="${BUILD_CONTAINER}"
        DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --runtime=runc"
        if [[ "${STAGE}" == "docs" ]]; then
            DOCKER_RUN_ARGS="${DOCKER_RUN_ARGS} --env MERGE_DOCS_YAML=1"
        fi
    fi

    if [[ "${STAGE}" == "bash" ]]; then
        DOCKER_RUN_CMD="bash --init-file /ci_tmp/bootstrap_local_ci.sh"
    else
        DOCKER_RUN_CMD="/ci_tmp/bootstrap_local_ci.sh"
    fi

    docker run ${DOCKER_RUN_ARGS} ${DOCKER_EXTRA_ARGS} ${CONTAINER} ${DOCKER_RUN_CMD}
done
