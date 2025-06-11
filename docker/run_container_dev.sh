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

export DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-"morpheus"}
export DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"dev-$(date +'%y%m%d')"}

export DOCKER_ARGS="-v $PWD:/workspace -v /dev/hugepages:/dev/hugepages --privileged"

# Call the general run script
${SCRIPT_DIR}/run_container.sh
