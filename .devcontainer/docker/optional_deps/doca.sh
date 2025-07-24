#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}

DOCA_OS_VERSION=${DOCA_OS_VERSION:-"22.04"}
DOCA_VERSION=${DOCA_VERSION:-2.7.0}

REAL_ARCH=${REAL_ARCH:-$(arch)}
if [[ ${REAL_ARCH} == "x86_64" ]]; then
    DOCA_ARCH="x86_64"
elif [[ ${REAL_ARCH} == "aarch64" ]]; then
    DOCA_ARCH="arm64-sbsa"
else
    echo "Unsupported architecture: ${REAL_ARCH}"
    exit 1
fi

DOCA_URL="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}/${LINUX_DISTRO}${DOCA_OS_VERSION}/${DOCA_ARCH}/"
DOCA_GPG_URL="https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub"

# Exit early if nothing to do
if [[ ${MORPHEUS_SUPPORT_DOCA} != @(TRUE|ON) ]]; then
   exit 0
fi

WORKING_DIR=$1
mkdir -p ${WORKING_DIR}
echo "Installing DOCA using directory: ${WORKING_DIR}"


echo "Adding DOCA repo: ${DOCA_URL}"
curl ${DOCA_GPG_URL} | gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" > /etc/apt/sources.list.d/doca.list

apt update

# Need to explicitly install the version of mft provided by the DOCA repo overriding the verdion from the cuda repo
# to avoid version conflicts.
# If/when we update either the OS, DOCA or CUDA version, we need to update the mft version here as well by checking
# the output of `apt policy mft`
apt install -y doca-all doca-gpu doca-gpu-dev mft=4.28.0-92

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
