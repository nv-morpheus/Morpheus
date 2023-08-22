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
/usr/bin/nvidia-smi

update_conda_env

rapids-logger "Check versions"
python3 --version
x86_64-conda-linux-gnu-cc --version
x86_64-conda-linux-gnu-c++ --version
cmake --version
ninja --version
sccache --version

git submodule update --init --recursive

CMAKE_FLAGS="${CMAKE_BUILD_ALL_FEATURES}"
CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_STUBS=ON"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_BUILD_WHEEL=OFF"
CMAKE_FLAGS="${CMAKE_FLAGS} -DMORPHEUS_PYTHON_PERFORM_INSTALL=ON"
if [[ "${LOCAL_CI}" == "" ]]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DCCACHE_PROGRAM_PATH=$(which sccache)"
fi

rapids-logger "Configuring cmake for Morpheus with ${CMAKE_FLAGS}"
cmake -B build -G Ninja ${CMAKE_FLAGS} .

rapids-logger "Building Morpheus"
cmake --build build --parallel ${PARALLEL_LEVEL}

if [[ "${LOCAL_CI}" == "" ]]; then
    rapids-logger "sccache usage for morpheus build:"
    sccache --show-stats
fi

rapids-logger "Checking Python stub files"

# Check for git diffs which would mean the build is out of sync with the repo
if [[ $(git status --short --untracked | grep .pyi) != "" ]]; then

    echo "The Python stubs (*.pyi) are out of sync with repo. Please rerun the build locally with "
    echo "'-DMORPHEUS_PYTHON_BUILD_STUBS=ON' and commit the stub files into the repo"
    git status
    exit 1
fi

CPP_TESTS=($(find ${MORPHEUS_ROOT}/build -name "*.x"))

rapids-logger "Pulling LFS assets"

git lfs install
${MORPHEUS_ROOT}/scripts/fetch_data.py fetch tests validation

# List missing files
rapids-logger "Listing missing files"
git lfs ls-files

REPORTS_DIR="${WORKSPACE_TMP}/reports"
mkdir -p ${WORKSPACE_TMP}/reports

rapids-logger "Running C++ tests"
# Running the tests from the tests dir. Normally this isn't nescesary, however since
# we are testing the installed version of morpheus in site-packages and not the one
# in the repo dir, the pytest coverage module reports incorrect coverage stats.
pushd ${MORPHEUS_ROOT}/tests

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
set +e

python -I -m pytest --run_slow --run_kafka --fail_missing \
       --junit-xml=${REPORTS_DIR}/report_pytest.xml \
       --cov=morpheus \
       --cov-report term-missing \
       --cov-report=xml:${REPORTS_DIR}/report_pytest_coverage.xml

PYTEST_RESULTS=$?
TEST_RESULTS=$(($TEST_RESULTS+$PYTEST_RESULTS))

set -e
popd

rapids-logger "Archiving test reports"
cd $(dirname ${REPORTS_DIR})
tar cfj ${WORKSPACE_TMP}/test_reports.tar.bz $(basename ${REPORTS_DIR})

rapids-logger "Pushing results to ${DISPLAY_ARTIFACT_URL}"
set_job_summary_preamble
upload_artifact ${WORKSPACE_TMP}/test_reports.tar.bz

exit ${TEST_RESULTS}
