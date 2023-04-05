#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -x
set -e

# Optionally can set INSTALL_PREFIX to build and install to a specific directory. Also causes cmake install to run
BUILD_DIR=${BUILD_DIR:-"build"}

echo "Runing CMake configure..."
cmake -B ${BUILD_DIR} -GNinja \
   -DCMAKE_MESSAGE_CONTEXT_SHOW=ON \
   -DMORPHEUS_USE_CLANG_TIDY=OFF \
   -DMORPHEUS_PYTHON_INPLACE_BUILD=ON \
   -DMORPHEUS_USE_CCACHE=ON \
   -DMORPHEUS_USE_CONDA=${MORPHEUS_USE_CONDA:-"ON"} \
   -DMRC_VERSION="23.07" \
   ${INSTALL_PREFIX:+"-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"} \
   ${CMAKE_CONFIGURE_EXTRA_ARGS:+"${CMAKE_CONFIGURE_EXTRA_ARGS}"} .

echo "Running CMake build..."
cmake --build ${BUILD_DIR} -j ${INSTALL_PREFIX:+"--target install"} "$@"
