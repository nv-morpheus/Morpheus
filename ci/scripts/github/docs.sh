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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

update_conda_env env.yaml

download_artifact "wheel.tar.bz"

tar xf "${WORKSPACE_TMP}/wheel.tar.bz"

pip install ${MORPHEUS_ROOT}/build/dist/*.whl

rapids-logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch docs examples

git submodule update --init --recursive

rapids-logger "Configuring for docs"
cmake -B build -G Ninja ${CMAKE_BUILD_ALL_FEATURES} -DMORPHEUS_PYTHON_BUILD_STUBS=OFF -DMORPHEUS_BUILD_DOCS=ON .

rapids-logger "Building docs"
cmake --build build --target morpheus_docs

rapids-logger "Archiving the docs"
tar cfj "${WORKSPACE_TMP}/docs.tar.bz" build/docs/html

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact "${WORKSPACE_TMP}/docs.tar.bz"

rapids-logger "Success"
exit 0
