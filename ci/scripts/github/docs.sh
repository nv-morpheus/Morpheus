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

set -e

source ${WORKSPACE}/ci/scripts/github/common.sh

update_conda_env

aws s3 cp --no-progress "${ARTIFACT_URL}/wheel.tar.bz" "${WORKSPACE_TMP}/wheel.tar.bz"

tar xf "${WORKSPACE_TMP}/wheel.tar.bz"

pip install ${MORPHEUS_ROOT}/build/wheel

rapids-logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch docs

cd ${MORPHEUS_ROOT}/docs
rapids-logger "Installing Documentation dependencies"
pip install -r requirement.txt

rapids-logger "Building docs"

# Work-around for dlopen failing on libcuda.so
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VER}/compat:${LD_LIBRARY_PATH}

make -j ${PARALLEL_LEVEL} html

rapids-logger "Tarring the docs"
tar cfj "${WORKSPACE_TMP}/docs.tar.bz" build/html

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${WORKSPACE_TMP}/docs.tar.bz" "${ARTIFACT_URL}/docs.tar.bz"

rapids-logger "Success"
exit 0
