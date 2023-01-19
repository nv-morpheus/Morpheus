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


set(MORPHEUS_RAPIDS_VERSION "22.08" CACHE STRING "Global default version for all Rapids project dependencies")
set(RAPIDS_CMAKE_VERSION "${MORPHEUS_RAPIDS_VERSION}" CACHE STRING "Version of rapids-cmake to use")

# Download and load the repo according to the rapids-cmake instructions if it does not exist
# NOTE: Use a different file than RAPIDS.cmake because MatX will just override it: https://github.com/NVIDIA/MatX/pull/192
if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS_CMAKE.cmake)
   message(STATUS "Downloading RAPIDS CMake Version: ${RAPIDS_CMAKE_VERSION}")
   file(
      DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_CMAKE_VERSION}/RAPIDS.cmake
      ${CMAKE_BINARY_DIR}/RAPIDS_CMAKE.cmake
   )
endif()

# Now load the file
include(${CMAKE_BINARY_DIR}/RAPIDS_CMAKE.cmake)

# Load Rapids Cmake packages
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
