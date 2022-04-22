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
export CCACHE_BASEDIR=$(realpath ${PWD}/..)
export CCACHE_LOGFILE=${CCACHE_DIR}/ccache.log
export CCACHE_DEBUG=1
export CCACHE_DEBUGDIR=${PWD}/ccache_debug
export CCACHE_SLOPPINESS="system_headers"

# Use the GNU paths to help ccache
export CC=${GCC}
export CXX=${GXX}

# CMake with nvcc uses -isystem=/path instead of -isystem /path which ccache doesnt like. Replace that
REPLACE_ISYSTEM="ARGS=()\nfor i in \"\${@}\"; do\n  ARGS+=(\${i/\"-isystem=/\"/\"-isystem /\"})\ndone\n"

# Setup using CCACHE
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_C_COMPILER_LAUNCHER} \"\${ARGS[@]}\"" > ccache_cc.sh
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_CXX_COMPILER_LAUNCHER} \"\${ARGS[@]}\"" > ccache_cxx.sh
echo -e '#!/bin/bash\n'"${REPLACE_ISYSTEM}\n${CMAKE_CUDA_COMPILER_LAUNCHER} \"\${ARGS[@]}\"" > ccache_cuda.sh

export CMAKE_C_COMPILER_LAUNCHER="${PWD}/ccache_cc.sh"
export CMAKE_CXX_COMPILER_LAUNCHER="${PWD}/ccache_cxx.sh"
export CMAKE_CUDA_COMPILER_LAUNCHER="${PWD}/ccache_cuda.sh"

chmod +x ${CMAKE_C_COMPILER_LAUNCHER}
chmod +x ${CMAKE_CXX_COMPILER_LAUNCHER}
chmod +x ${CMAKE_CUDA_COMPILER_LAUNCHER}

# Fix __FILE__ macros that break caching and include the full prefixed path (very long)
export CFLAGS="${CXXFLAGS} -fmacro-prefix-map=${PREFIX}=/usr/local/src/conda-prefix"
export CXXFLAGS="${CXXFLAGS} -fmacro-prefix-map=${PREFIX}=/usr/local/src/conda-prefix"
export CUDAFLAGS="${CUDAFLAGS} -fmacro-prefix-map=${PREFIX}=/usr/local/src/conda-prefix"

echo "=====Printing Env====="
./print_env.sh

echo "=====Cleaning====="
./build.sh clean

echo "=====Building====="
if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    # This assumes the script is executed from the root of the repo directory
    # ./build.sh libcudf --allgpuarch --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\" --ptds
    ./build.sh libcudf --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\" --ptds
else
    # ./build.sh libcudf --allgpuarch --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\" --ptds
    ./build.sh libcudf --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\" --ptds
fi
