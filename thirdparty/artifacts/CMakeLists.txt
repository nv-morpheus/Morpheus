# SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
list(APPEND CMAKE_MESSAGE_CONTEXT "thirdparty")

option(MORPHEUS_TP_BUILD_DOCA_DEPS "Configure Morpheus to build with NVIDIA DOCA" OFF)
option(MORPHEUS_TP_BUILD_LIBMD "Build third party library libmd" OFF)
option(MORPHEUS_TP_BUILD_LIBBSD "Build third party library libbsd" OFF)
option(MORPHEUS_TP_BUILD_LIBGDRCOPY "Build third party library gdrcopy" OFF)

if (MORPHEUS_TP_BUILD_DOCA_DEPS)
  set(MORPHEUS_TP_BUILD_LIBMD "ON")
  set(MORPHEUS_TP_BUILD_LIBBSD "ON")
  set(MORPHEUS_TP_BUILD_GDRCOPY "ON")
endif()

option(MORPHEUS_TP_INSTALL_DOCA_DEPS "Install DOCA dependencies into local system (primarily for Docker)" OFF)

set(MORPHEUS_TP_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MORPHEUS_TP_CACHE_DIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Add the RAPIDS cmake helper scripts
include(cmake/setup_rapids_cmake.cmake)

rapids_cmake_build_type(Release)

# Add conda environment to path
include(cmake/setup_conda_toolchain.cmake)

project(srf_third_party_libs LANGUAGES C CXX)

# Needed force rapids cpm to use our source directory.
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache/cpm")
include(setup_cpm)

add_subdirectory(libraries)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)