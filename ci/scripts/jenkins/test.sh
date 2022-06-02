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

source ${WORKSPACE}/ci/scripts/jenkins/common.sh
/usr/bin/nvidia-smi

gpuci_logger "Check versions"
python3 --version
gcc --version
g++ --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

gpuci_logger "Downloading build artifacts from ${DISPLAY_ARTIFACT_URL}"
aws s3 cp --no-progress "${ARTIFACT_URL}/conda_env.tar.gz" "${WORKSPACE_TMP}/conda_env.tar.gz"
aws s3 cp --no-progress "${ARTIFACT_URL}/workspace.tar.bz" "${WORKSPACE_TMP}/workspace.tar.bz"

gpuci_logger "Extracting"
mkdir -p /opt/conda/envs/morpheus
tar xf "${WORKSPACE_TMP}/conda_env.tar.gz" --no-same-owner --directory /opt/conda/envs/morpheus
tar xf "${WORKSPACE_TMP}/workspace.tar.bz" --no-same-owner

gpuci_logger "Setting test env"
conda activate morpheus
conda-unpack
conda list --show-channel-urls

npm install --slient -g camouflage-server
mamba install -q -y -c conda-forge "git-lfs=3.1.4"

gpuci_logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}
git lfs install
${MORPHEUS_ROOT}/ci/scripts/fetch_data.py tests

pip install -e ${MORPHEUS_ROOT}

# Work-around for issue where libmorpheus_utils.so is not found by libmorpheus.so
# The build and test nodes have different workspace paths (/jenkins vs. /var/lib/jenkins)
# Typically these are fixed by conda-unpack but since we did an in-place build we will
# have to fix this ourselves by setting LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MORPHEUS_ROOT}/morpheus/_lib

gpuci_logger "Running tests"
set +e
pytest --run_slow \
       --junit-xml=${WORKSPACE_TMP}/report_pytest.xml \
       --cov=morpheus \
       --cov-report term-missing \
       --cov-report=xml:${WORKSPACE_TMP}/report_pytest_coverage.xml

PYTEST_RESULTS=$?
set -e

gpuci_logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp ${WORKSPACE_TMP}/report_pytest.xml "${ARTIFACT_URL}/report_pytest.xml"
aws s3 cp ${WORKSPACE_TMP}/report_pytest_coverage.xml "${ARTIFACT_URL}/report_pytest_coverage.xml"

exit ${PYTEST_RESULTS}
