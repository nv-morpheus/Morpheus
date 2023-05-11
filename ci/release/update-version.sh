#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Usage
# Either supply full versions:
#    `bash update-version.sh <current_version> <new_version>`
#    Format is YY.MM.PP - no leading 'v' or trailing 'a'
# Or no versions:
#    `bash update-version.sh`

set -e

# If the user has not supplied the versions, determine them from the git tags
if [[ "$#" -ne 2 ]]; then
   echo "No versions were provided. Using last 2 git tags to determined current and next version"

   # Current version comes from the previous alpha tag
   CURRENT_FULL_VERSION=$(git tag --merged HEAD --list 'v*' | sort --version-sort | tail -n 2 | head -n 1 | tr -d 'va')
   # Next version comes from the latest alpha tag
   NEXT_FULL_VERSION=$(git tag --merged HEAD --list 'v*' | sort --version-sort | tail -n 1 | tr -d 'va')
else
   # User has supplied current and next arguments
   CURRENT_FULL_VERSION=$1
   NEXT_FULL_VERSION=$2
fi

CURRENT_MAJOR=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo ${CURRENT_FULL_VERSION} | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

NEXT_MAJOR=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[2]}')
NEXT_PATCH=$(echo ${NEXT_FULL_VERSION} | awk '{split($0, a, "."); print a[3]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Need to distutils-normalize the versions for some use cases
CURRENT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${CURRENT_SHORT_TAG}'))")
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

echo "Preparing release $CURRENT_FULL_VERSION (PEP ${CURRENT_SHORT_TAG_PEP440}) => $NEXT_FULL_VERSION (PEP ${NEXT_SHORT_TAG_PEP440})"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# .gitmodules
git submodule set-branch -b branch-${NEXT_SHORT_TAG} external/morpheus-visualizations
git submodule set-branch -b branch-${NEXT_SHORT_TAG} morpheus_utils

# Root CMakeLists.txt
sed_runner 's/'"VERSION ${CURRENT_FULL_VERSION}.*"'/'"VERSION ${NEXT_FULL_VERSION}"'/g' CMakeLists.txt

# Conda environment file
sed_runner "s/mrc=${CURRENT_SHORT_TAG}/mrc=${NEXT_SHORT_TAG}/g" docker/conda/environments/cuda11.8_dev.yml

# examples/digital_fingerprinting
sed_runner "s/v${CURRENT_FULL_VERSION}-runtime/v${NEXT_FULL_VERSION}-runtime/g" examples/digital_fingerprinting/production/docker-compose.yml
sed_runner "s/v${CURRENT_FULL_VERSION}-runtime/v${NEXT_FULL_VERSION}-runtime/g" examples/digital_fingerprinting/production/Dockerfile
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" examples/digital_fingerprinting/starter/README.md

# docs/source/cloud_deployment_guide.md
sed_runner "s|${CURRENT_SHORT_TAG}.tgz|${NEXT_SHORT_TAG}.tgz|g" docs/source/cloud_deployment_guide.md
sed_runner "s|blob/branch-${CURRENT_SHORT_TAG}|blob/branch-${NEXT_SHORT_TAG}|g" docs/source/cloud_deployment_guide.md
sed_runner "s|tree/branch-${CURRENT_SHORT_TAG}|tree/branch-${NEXT_SHORT_TAG}|g" docs/source/cloud_deployment_guide.md

# docs/source/getting_started.md
# Only do the minor version here since the full version can mess up the examples
sed_runner "s/${CURRENT_SHORT_TAG}/${NEXT_SHORT_TAG}/g" docs/source/getting_started.md
