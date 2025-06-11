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

set -e +o pipefail
# set -x
# set -v

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure to load the utils and globals
source ${SCRIPT_DIR}/val-globals.sh
source ${SCRIPT_DIR}/val-utils.sh

# Load triton here to prevent it being closed and restarted after each command
ensure_triton_running

# Run everything once USE_CPP=True
export USE_CPP=1

${SCRIPT_DIR}/abp/val-abp-all.sh

# Disabled per #1641
# ${SCRIPT_DIR}/hammah/val-hammah-all.sh

${SCRIPT_DIR}/phishing/val-phishing-all.sh
${SCRIPT_DIR}/sid/val-sid-all.sh
