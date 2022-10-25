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
/usr/bin/nvidia-smi

install_deb_deps

# Restore the environment and then ensure we have the CI dependencies
restore_conda_env

# Install the built Morpheus python package
pip install ${MORPHEUS_ROOT}/build/wheel

CPP_TESTS=($(find ${MORPHEUS_ROOT}/build/wheel -name "*.x"))

rapids-logger "Installing test dependencies"
npm install --silent -g camouflage-server

# Kafka tests need Java, since this stage is the only one that needs it, installing it here rather than adding it to
# the ci.yaml file
mamba install -c conda-forge "openjdk=11.0.15"
export PYTEST_KAFKA_DIR=${WORKSPACE_TMP}/pytest-kafka

# Installing pytest-kafka from source instead of conda/pip as the setup.py includes helper methods for downloading Kafka
# https://gitlab.com/karolinepauls/pytest-kafka/-/issues/9
git clone https://gitlab.com/karolinepauls/pytest-kafka.git ${PYTEST_KAFKA_DIR}
pushd ${PYTEST_KAFKA_DIR}
python setup.py develop
popd

rapids-logger "Pulling LFS assets"
cd ${MORPHEUS_ROOT}

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch tests validation

# List missing files
rapids-logger "Listing missing files"
git lfs ls-files

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

TEST_RESULTS=0
for cpp_test in "${CPP_TESTS[@]}"; do
       test_name=$(basename ${cpp_test})
       rapids-logger "Running ${test_name}"
       set +e

       ${cpp_test} --gtest_output="xml:${REPORTS_DIR}/report_${test_name}.xml"
       TEST_RESULT=$?
       TEST_RESULTS=$(($TEST_RESULTS+$TEST_RESULT))

       set -e
done

rapids-logger "Running Python tests"
# Running the tests from the tests dir. Normally this isn't nescesary, however since
# we are testing the installed version of morpheus in site-packages and not the one
# in the repo dir, the pytest coverage module reports incorrect coverage stats.
cd ${MORPHEUS_ROOT}/tests

set +e

python -I -m pytest --run_slow --run_kafka \
       --junit-xml=${REPORTS_DIR}/report_pytest.xml \
       --cov=morpheus \
       --cov-report term-missing \
       --cov-report=xml:${REPORTS_DIR}/report_pytest_coverage.xml

PYTEST_RESULTS=$?
TEST_RESULTS=$(($TEST_RESULTS+$PYTEST_RESULTS))

set -e

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
aws s3 cp ${WORKSPACE_TMP}/test_reports.tar.bz "${ARTIFACT_URL}/test_reports.tar.bz"

exit ${TEST_RESULTS}
