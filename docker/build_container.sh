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


DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:?"Must set \$DOCKER_IMAGE_NAME to build. Use the dev/release scripts to set these automatically"}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:?"Must set \DOCKER_IMAGE_TAG to build. Use the dev/release scripts to set these automatically"}
DOCKER_TARGET=${DOCKER_TARGET:-"runtime"}
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
DOCKER_EXTRA_ARGS=${DOCKER_EXTRA_ARGS:-""}

# Build args
FROM_IMAGE=${FROM_IMAGE:-"rapidsai/mambaforge-cuda"}
CUDA_MAJOR_VER=${CUDA_MAJOR_VER:-11}
CUDA_MINOR_VER=${CUDA_MINOR_VER:-8}
CUDA_REV_VER=${CUDA_REV_VER:-0}
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-20.04}
RAPIDS_VER=${RAPIDS_VER:-23.02}
PYTHON_VER=${PYTHON_VER:-3.10}
TENSORRT_VERSION=${TENSORRT_VERSION:-8.2.1.3}

DOCKER_ARGS="-t ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
DOCKER_ARGS="${DOCKER_ARGS} --target ${DOCKER_TARGET}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg FROM_IMAGE=${FROM_IMAGE}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_MAJOR_VER=${CUDA_MAJOR_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_MINOR_VER=${CUDA_MINOR_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg CUDA_REV_VER=${CUDA_REV_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_DISTRO=${LINUX_DISTRO}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg LINUX_VER=${LINUX_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg RAPIDS_VER=${RAPIDS_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg PYTHON_VER=${PYTHON_VER}"
DOCKER_ARGS="${DOCKER_ARGS} --build-arg TENSORRT_VERSION=${TENSORRT_VERSION}"
DOCKER_ARGS="${DOCKER_ARGS} --network=host"

# Last add any extra args (duplicates override earlier ones)
DOCKER_ARGS="${DOCKER_ARGS} ${DOCKER_EXTRA_ARGS}"

# Export buildkit variable
export DOCKER_BUILDKIT=1

echo "Building morpheus:${DOCKER_TAG}..."
echo "   FROM_IMAGE      : ${FROM_IMAGE}"
echo "   CUDA_MAJOR_VER  : ${CUDA_MAJOR_VER}"
echo "   CUDA_MINOR_VER  : ${CUDA_MINOR_VER}"
echo "   CUDA_REV_VER    : ${CUDA_REV_VER}"
echo "   LINUX_DISTRO    : ${LINUX_DISTRO}"
echo "   LINUX_VER       : ${LINUX_VER}"
echo "   RAPIDS_VER      : ${RAPIDS_VER}"
echo "   PYTHON_VER      : ${PYTHON_VER}"
echo "   TENSORRT_VERSION: ${TENSORRT_VERSION}"
echo ""
echo "   COMMAND: docker build ${DOCKER_ARGS} -f docker/Dockerfile ."
echo "   Note: add '--progress plain' to DOCKER_ARGS to show all container build output"

docker build ${DOCKER_ARGS} -f docker/Dockerfile .
