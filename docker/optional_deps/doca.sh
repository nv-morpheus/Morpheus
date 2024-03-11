#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MORPHEUS_SUPPORT_DOCA=${MORPHEUS_SUPPORT_DOCA:-OFF}

# Exit early if nothing to do
if [[ ${MORPHEUS_SUPPORT_DOCA} != @(TRUE|ON) ]]; then
   exit 0
fi

DOCA_REPO_HOST=${DOCA_REPO_HOST:?"Must set \$DOCA_REPO_HOST to build DOCA."}
DOCA_VERSION=${DOCA_VERSION:-2.6.0-0.0.1}
WORKING_DIR=$1

echo "Installing DOCA using directory: ${WORKING_DIR}"

DEB_DIR=${WORKING_DIR}/deb

mkdir -p ${DEB_DIR}

# Download all files with -nc to skip download if its already there
wget -nc -P ${DEB_DIR} https://${DOCA_REPO_HOST}/doca-repo-2.6.0/doca-repo-2.6.0-0.0.1-240205-083002-daily/doca-host-repo-ubuntu2204_2.6.0-0.0.1-240205-083002-daily.2.6.0058.1.24.01.0.3.3.1_amd64.deb
# Install the doca host repo
dpkg -i ${DEB_DIR}/doca-host-repo*.deb

# Install all other packages
apt-get update
# apt-get install -y libjson-c-dev meson cmake pkg-config
apt-get install -y doca-sdk doca-runtime doca-gpu doca-gpu-dev

# Now install the gdrcopy library according to: https://github.com/NVIDIA/gdrcopy
GDRCOPY_DIR=${WORKING_DIR}/gdrcopy

if [[ ! -d "${GDRCOPY_DIR}" ]] ; then
    git clone https://github.com/NVIDIA/gdrcopy.git ${GDRCOPY_DIR}
    cd ${GDRCOPY_DIR}
else
    cd ${GDRCOPY_DIR}
    git pull https://github.com/NVIDIA/gdrcopy.git
fi

make lib lib_install
