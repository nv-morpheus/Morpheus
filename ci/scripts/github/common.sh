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

export CONDA_ENV_YML=${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_dev.yml
export CONDA_EXAMPLES_YML=${MORPHEUS_ROOT}/docker/conda/environments/cuda${CUDA_VER}_examples.yml
export CONDA_DOCS_YML=${MORPHEUS_ROOT}/docs/conda_docs.yml
export PIP_REQUIREMENTS=${MORPHEUS_ROOT}/docker/conda/environments/requirements.txt

export CMAKE_BUILD_ALL_FEATURES="-DCMAKE_MESSAGE_CONTEXT_SHOW=ON -DMORPHEUS_CUDA_ARCHITECTURES=60;70;75;80 -DMORPHEUS_BUILD_BENCHMARKS=ON -DMORPHEUS_BUILD_EXAMPLES=ON -DMORPHEUS_BUILD_TESTS=ON -DMORPHEUS_USE_CONDA=ON -DMORPHEUS_PYTHON_INPLACE_BUILD=OFF -DMORPHEUS_PYTHON_BUILD_STUBS=ON -DMORPHEUS_USE_CCACHE=ON"

export FETCH_STATUS=0

print_env_vars

function update_conda_env() {
    # Deactivate the environment first before updating
    conda deactivate

    ENV_YAML=${CONDA_ENV_YML}
    if [[ "${MERGE_EXAMPLES_YAML}" == "1" || "${MERGE_DOCS_YAML}" == "1" ]]; then
        # Merge the dev, docs and examples envs, otherwise --prune will remove the examples packages
        ENV_YAML=${condatmpdir}/merged_env.yml
        YAMLS="${CONDA_ENV_YML}"
        if [[ "${MERGE_EXAMPLES_YAML}" == "1" ]]; then
            YAMLS="${YAMLS} ${CONDA_EXAMPLES_YML}"
        fi
        if [[ "${MERGE_DOCS_YAML}" == "1" ]]; then
            YAMLS="${YAMLS} ${CONDA_DOCS_YML}"
        fi

        # Conda is going to expect a requirements.txt file to be in the same directory as the env yaml
        cp ${PIP_REQUIREMENTS} ${condatmpdir}/requirements.txt

        rapids-logger "Merging conda envs: ${YAMLS}"
        conda run -n morpheus --live-stream conda-merge ${YAMLS} > ${ENV_YAML}
    fi

    rapids-logger "Checking for updates to conda env"

    # Update the packages
    rapids-mamba-retry env update -n morpheus --prune -q --file ${ENV_YAML}

    # Finally, reactivate
    conda activate morpheus

    rapids-logger "Final Conda Environment"
    show_conda_info
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

function show_conda_info() {

    rapids-logger "Check Conda info"
    conda info
    conda config --show-sources
    conda list --show-channel-urls
}
