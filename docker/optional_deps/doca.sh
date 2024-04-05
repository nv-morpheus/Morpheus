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
LINUX_DISTRO=${LINUX_DISTRO:-ubuntu}
LINUX_VER=${LINUX_VER:-22.04}
DOCA_VERSION=${DOCA_VERSION:-2.6.0}

# Exit early if nothing to do
if [[ ${MORPHEUS_SUPPORT_DOCA} != @(TRUE|ON) ]]; then
   exit 0
fi

WORKING_DIR=$1

echo "Installing DOCA using directory: ${WORKING_DIR}"

DEB_DIR=${WORKING_DIR}/deb

mkdir -p ${DEB_DIR}

DOCA_REPO_LINK="https://linux.mellanox.com/public/repo/doca/${DOCA_VERSION}"
DOCA_REPO="${DOCA_REPO_LINK}/ubuntu22.04"
DOCA_REPO_ARCH="x86_64"
DOCA_UPSTREAM_REPO="${DOCA_REPO}/${DOCA_REPO_ARCH}"

# Upgrade the base packages (diff between image and Canonical upstream repo)
apt update -y
apt upgrade -y

# Cleanup apt
rm -rf /var/lib/apt/lists/*
apt autoremove -y

# Configure DOCA Repository, and install packages
apt update -y

# Install wget & Add the DOCA public repository
apt install -y --no-install-recommends wget software-properties-common gpg-agent
wget -qO - ${DOCA_UPSTREAM_REPO}/GPG-KEY-Mellanox.pub | apt-key add -
add-apt-repository "deb [trusted=yes] ${DOCA_UPSTREAM_REPO} ./"
apt update -y

# Install base-rt content
apt install -y --no-install-recommends \
    doca-gpu \
    doca-gpu-dev \
    doca-prime-runtime \
    doca-prime-sdk \
    doca-sdk \
    dpcp \
    flexio \
    ibacm \
    ibverbs-utils \
    librdmacm1 \
    libibnetdisc5 \
    libibumad3 \
    libibmad5 \
    libopensm \
    libopenvswitch \
    libyara8 \
    mlnx-tools \
    ofed-scripts \
    openmpi \
    openvswitch-common \
    openvswitch-switch \
    srptools \
    mlnx-ethtool \
    mlnx-iproute2 \
    python3-pyverbs \
    rdma-core \
    ucx \
    yara

    # Cleanup apt
rm -rf /usr/lib/python3/dist-packages
apt remove -y software-properties-common gpg-agent
rm -rf /var/lib/apt/lists/*
apt autoremove -y

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
