#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function print_env_vars() {
    rapids-logger "Environ:"
    env | grep -v -E "AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|GH_TOKEN" | sort
}

rapids-logger "Env Setup"
print_env_vars
rapids-logger "---------"
mkdir -p ${WORKSPACE_TMP}
source /opt/conda/etc/profile.d/conda.sh
export MORPHEUS_ROOT=${MORPHEUS_ROOT:-$(git rev-parse --show-toplevel)}
cd ${MORPHEUS_ROOT}

# For non-gpu hosts nproc will correctly report the number of cores we are able to use
# On a GPU host however nproc will report the total number of cores and PARALLEL_LEVEL
# will be defined specifying the subset we are allowed to use.
NUM_CORES=$(nproc)
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-${NUM_CORES}}
rapids-logger "Procs: ${NUM_CORES}"
/usr/bin/lscpu

rapids-logger "Memory"
/usr/bin/free -g

rapids-logger "User Info"
id

# For PRs, $GIT_BRANCH is like: pull-request/989
REPO_NAME=$(basename "${GITHUB_REPOSITORY}")
ORG_NAME="${GITHUB_REPOSITORY_OWNER}"
PR_NUM="${GITHUB_REF_NAME##*/}"

# S3 vars
export S3_URL="s3://rapids-downloads/ci/morpheus"
export DISPLAY_URL="https://downloads.rapids.ai/ci/morpheus"
export ARTIFACT_ENDPOINT="/pull-request/${PR_NUM}/${GIT_COMMIT}/${NVARCH}"
export ARTIFACT_URL="${S3_URL}${ARTIFACT_ENDPOINT}"
export DISPLAY_ARTIFACT_URL="${DISPLAY_URL}${ARTIFACT_ENDPOINT}/"

# Set sccache env vars
export SCCACHE_S3_KEY_PREFIX=morpheus-${NVARCH}
export SCCACHE_BUCKET=rapids-sccache-east
export SCCACHE_REGION="us-east-2"
export SCCACHE_IDLE_TIMEOUT=32768
#export SCCACHE_LOG=debug

# Work-around for https://github.com/rapidsai/cudf/issues/12862
export CUDACXX=${CUDA_HOME}/bin/nvcc

export CMAKE_BUILD_ALL_FEATURES="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMORPHEUS_CUDA_ARCHITECTURES=60;70;75;80 -DMORPHEUS_BUILD_BENCHMARKS=ON -DMORPHEUS_BUILD_EXAMPLES=ON -DMORPHEUS_BUILD_TESTS=ON -DMORPHEUS_USE_CONDA=ON -DMORPHEUS_PYTHON_INPLACE_BUILD=OFF -DMORPHEUS_PYTHON_BUILD_STUBS=ON -DMORPHEUS_USE_CCACHE=ON"

export FETCH_STATUS=0

print_env_vars

function update_conda_env() {
    rapids-logger "Checking for updates to conda env"
    rapids-mamba-retry env update -n morpheus -q --file ${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_dev.yml
    conda deactivate
    conda activate morpheus
}


function fetch_base_branch() {
    rapids-logger "Retrieving base branch from GitHub API"
    [[ -n "$GH_TOKEN" ]] && CURL_HEADERS=('-H' "Authorization: token ${GH_TOKEN}")
    RESP=$(
    curl -s \
        -H "Accept: application/vnd.github.v3+json" \
        "${CURL_HEADERS[@]}" \
        "${GITHUB_API_URL}/repos/${ORG_NAME}/${REPO_NAME}/pulls/${PR_NUM}"
    )

    BASE_BRANCH=$(echo "${RESP}" | jq -r '.base.ref')

    # Change target is the branch name we are merging into but due to the weird way jenkins does
    # the checkout it isn't recognized by git without the origin/ prefix
    export CHANGE_TARGET="origin/${BASE_BRANCH}"
    rapids-logger "Base branch: ${BASE_BRANCH}"
}

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

    rapids-logger "Downloading build artifacts from ${DISPLAY_ARTIFACT_URL}"
    fetch_s3 "${ARTIFACT_ENDPOINT}/conda_env.tar.gz" "${WORKSPACE_TMP}/conda_env.tar.gz"
    fetch_s3 "${ARTIFACT_ENDPOINT}/wheel.tar.bz" "${WORKSPACE_TMP}/wheel.tar.bz"

    rapids-logger "Extracting"
    mkdir -p /opt/conda/envs/morpheus

    # We are using the --no-same-owner flag since user id & group id's are inconsistent between nodes in our CI pool
    tar xf "${WORKSPACE_TMP}/conda_env.tar.gz" --no-same-owner --directory /opt/conda/envs/morpheus
    tar xf "${WORKSPACE_TMP}/wheel.tar.bz" --no-same-owner --directory ${MORPHEUS_ROOT}

    rapids-logger "Setting conda env"
    conda activate morpheus
    conda-unpack
}

function show_conda_info() {

    rapids-logger "Check Conda info"
    conda info
    conda config --show-sources
    conda list --show-channel-urls
}
