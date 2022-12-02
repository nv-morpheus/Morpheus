# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus")

option(MORPHEUS_PYTHON_INPLACE_BUILD
        "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)
option(MORPHEUS_BUILD_PYTHON_STUBS
        "Whether or not to generate stubs for python files." ON)

# TODO(Devin): Not currently using this, just adding to make module and lib creation
morpheus_utils_create_python_package(morpheus-package)

# Save the root of the python for relative paths
set(MORPHEUS_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Determine the install directory for built modules
file(RELATIVE_PATH python_relative_install ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} ${Python3_SITELIB})

# Find dependencies
find_package(CUDAToolkit REQUIRED)

set(CYTHON_FLAGS "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True" CACHE STRING "The directives for Cython compilation.")

add_subdirectory(_lib)


list(POP_BACK CMAKE_MESSAGE_CONTEXT)
