# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# Use the parent folder of work for the base dir to capture host and env files
export CCACHE_BASEDIR=$(realpath ${SRC_DIR}/..)
export CCACHE_LOGFILE=${CCACHE_DIR}/ccache.log
export CCACHE_DEBUG=1
export CCACHE_DEBUGDIR=${SRC_DIR}/ccache_debug
export CCACHE_SLOPPINESS="system_headers"
export CCACHE_NOHASHDIR=1

# Double check that the cache dir has been created
mkdir -p ${CCACHE_DIR}

# CMake with nvcc uses -isystem=/path instead of -isystem /path which ccache doesnt like. Replace that
REPLACE_ISYSTEM="ARGS=()\nfor i in \"\${@}\"; do\n  ARGS+=(\${i/\"-isystem=/\"/\"-isystem /\"})\ndone\n"

# Setup using CCACHE
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_C_COMPILER_LAUNCHER} ${GCC} \"\${ARGS[@]}\"" > ccache_cc.sh
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_CXX_COMPILER_LAUNCHER} ${GXX} \"\${ARGS[@]}\"" > ccache_cxx.sh
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_CUDA_COMPILER_LAUNCHER} nvcc \"\${ARGS[@]}\"" > ccache_cuda.sh

# For some reason CXX must be a single executable (i.e. so that `which $CXX` would not error). So instead of setting
# CXX="ccache $CXX", make a new script to do this for us
export CC="${PWD}/ccache_cc.sh"
export CXX="${PWD}/ccache_cxx.sh"
export NVCC="${PWD}/ccache_cuda.sh"

chmod +x ${CC}
chmod +x ${CXX}
chmod +x ${NVCC}

./print_env.sh

# This assumes the script is executed from the root of the repo directory
./build.sh cudf --ptds
