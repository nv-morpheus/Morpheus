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

gpuci_logger "Env Setup"
gpuci_logger "Environ:"
env | sort
gpuci_logger "---------"
mkdir -p ${WORKSPACE_TMP}
source /opt/conda/etc/profile.d/conda.sh
export MORPHEUS_ROOT=${MORPHEUS_ROOT:-$(git rev-parse --show-toplevel)}
echo "cur_dir=$(pwd) mr=${MORPHEUS_ROOT}"
cd ${MORPHEUS_ROOT}

# For non-gpu hosts nproc will correctly report the number of cores we are able to use
# On a GPU host however nproc will report the total number of cores and PARALLEL_LEVEL
# will be defined specifying the subset we are allowed to use.
NUM_CORES=$(nproc)
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-${NUM_CORES}}
gpuci_logger "Procs: ${NUM_CORES}"
/usr/bin/lscpu

gpuci_logger "Memory"
/usr/bin/free -g

gpuci_logger "User Info"
id

# For PRs, $GIT_BRANCH is like: pull-request/989
PR_NUM="${GIT_BRANCH##*/}"

# S3 vars
export S3_URL="s3://rapids-downloads/ci/morpheus"
export DISPLAY_URL="https://downloads.rapids.ai/ci/morpheus"
export ARTIFACT_ENDPOINT="/pull-request/${PR_NUM}/${GIT_COMMIT}/${NVARCH}"
export ARTIFACT_URL="${S3_URL}${ARTIFACT_ENDPOINT}"
export DISPLAY_ARTIFACT_URL="${DISPLAY_URL}/pull-request/${PR_NUM}/${GIT_COMMIT}/${NVARCH}/"

# Set sccache env vars
export SCCACHE_S3_KEY_PREFIX=morpheus-${NVARCH}
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768
#export SCCACHE_LOG=debug

export CMAKE_BUILD_ALL_FEATURES="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMORPHEUS_BUILD_BENCHMARKS=ON -DMORPHEUS_BUILD_EXAMPLES=ON -DMORPHEUS_BUILD_TESTS=ON -DMORPHEUS_USE_CONDA=ON -DMORPHEUS_PYTHON_INPLACE_BUILD=OFF -DMORPHEUS_USE_CCACHE=ON"

export FETCH_STATUS=0

gpuci_logger "Environ:"
env | sort

function fetch_s3() {
    ENDPOINT=$1
    DESTINATION=$2
    if [[ "${USE_S3_CURL}" == "1" ]]; then
        curl -f "${DISPLAY_URL}${ENDPOINT}" -o "${DESTINATION}"
        FETCH_STATUS=$?
    else
        aws s3 cp --no-progress "${S3_URL}${ENDPOINT}" "${DESTINATION}"
        FETCH_STATUS=$?
    fi
}

function restore_conda_env() {

    gpuci_logger "Downloading build artifacts from ${DISPLAY_ARTIFACT_URL}"
    fetch_s3 "${ARTIFACT_ENDPOINT}/conda_env.tar.gz" "${WORKSPACE_TMP}/conda_env.tar.gz"
    fetch_s3 "${ARTIFACT_ENDPOINT}/wheel.tar.bz" "${WORKSPACE_TMP}/wheel.tar.bz"

    gpuci_logger "Extracting"
    mkdir -p /opt/conda/envs/morpheus

    # We are using the --no-same-owner flag since user id & group id's are inconsistent between nodes in our CI pool
    tar xf "${WORKSPACE_TMP}/conda_env.tar.gz" --no-same-owner --directory /opt/conda/envs/morpheus
    tar xf "${WORKSPACE_TMP}/wheel.tar.bz" --no-same-owner --directory ${MORPHEUS_ROOT}

    gpuci_logger "Setting conda env"
    conda activate morpheus
    conda-unpack
}

function show_conda_info() {

    gpuci_logger "Check Conda info"
    conda info
    conda config --show-sources
    conda list --show-channel-urls
}
