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

source ci/scripts/jenkins_common.sh
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
gpuci_logger "Packages installed in morpheus env"
conda list --show-channel-urls

echo "dir info"
ls -ld $(dirname ${CONDA_PREFIX})
ls -ld ${CONDA_PREFIX}
ls -ld ${CONDA_PREFIX}/lib
ls -ld ${CONDA_PREFIX}/lib/node_modules

echo "installing test packages"
which npm

npm root -g
echo ${HOME}
npm install --slient -g camouflage-server
echo "Installing git-lfs"
mamba install -q -y -c conda-forge "git-lfs=3.1.4"

gpuci_logger "Pulling LFS assets"
git lfs install
git lfs pull

pip install -e ${MORPHEUS_ROOT}

# Work-around for issue where libmorpheus_utils.so is not found by libmorpheus.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WORKSPACE}/morpheus/_lib

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
