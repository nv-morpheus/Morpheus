# SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Conda / Vcpkg configuration
list(APPEND CMAKE_MESSAGE_CONTEXT "toolchain")
message(STATUS "Configuring Toolchain Components")

if(DEFINED CMAKE_TOOLCHAIN_FILE)
  message(STATUS "[CMAKE_TOOLCHAIN_FILE] is defined (${CMAKE_TOOLCHAIN_FILE}), using custom toolchain")
  message(STATUS "Conda and VCPKG environment variables will be ignored.")
elseif (MORPHEUS_USE_CONDA AND DEFINED ENV{CONDA_PREFIX})
  message(STATUS "MORPHEUS_USE_CONDA is defined and CONDA environment ($ENV{CONDA_PREFIX}) exists.")
  message(STATUS "VCPKG environment variables will be ignored.")
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE STRING "" FORCE)
  endif()

  message(STATUS "Prepending CONDA_PREFIX ($ENV{CONDA_PREFIX}) to CMAKE_FIND_ROOT_PATH")
  list(PREPEND CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
  list(REMOVE_DUPLICATES CMAKE_FIND_ROOT_PATH)

elseif(DEFINED ENV{VCPKG_ROOT})
  if(NOT EXISTS "$ENV{VCPKG_ROOT}")
    message(FATAL_ERROR "Vcpkg env 'VCPKG_ROOT' set to '$ENV{VCPKG_ROOT}' but file does not exist! Exiting...")
    return()
  endif()
  message(STATUS "VCPKG_ROOT is defined and root directory ($ENV{VCPKG_ROOT}) exists.")
  message(STATUS "VCPKG will be used for morpheus build")

  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
  set(USING_VCPKG True)

  message(STATUS "Creating project. If this hangs, check 'VCPKG_ROOT' environment variable.
    Should not take more than a few seconds to see additional output")

  # If using shared libs (the default) use a custom triplet file to use dynamic linking
  if(BUILD_SHARED_LIBS)
    set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcpkg_triplets")
    set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic")
  endif()

  # Once the build type is set, remove any dumb vcpkg debug folders from the
  # search paths. Without this FindBoost fails since it defaults to the debug
  # binaries
  if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
    message(STATUS "Release Build: Removing debug paths from CMAKE_PREFIX_PATH and CMAKE_FIND_ROOT_PATH")
    list(REMOVE_ITEM CMAKE_PREFIX_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
    list(REMOVE_ITEM CMAKE_FIND_ROOT_PATH "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug")
  endif()

  # Help vcpkg out on CI systems by ensuring the cache directory exists
  if(DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE} AND NOT EXISTS "$ENV{VCPKG_DEFAULT_BINARY_CACHE}")
    message(STATUS "VCPKG binary cache missing. Creating directory. Cache location: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
    file(MAKE_DIRECTORY "$ENV{VCPKG_DEFAULT_BINARY_CACHE}")
  endif()
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
