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
$(restore_conda_env)

npm install --silent -g camouflage-server
mamba install -q -y -c conda-forge "git-lfs=3.1.4"

gpuci_logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}
git lfs install
git lfs pull

gpuci_logger "Running tests"
set +e
python -I -m pytest --run_slow \
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
