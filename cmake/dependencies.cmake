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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

# Needed force rapids cpm to use our source directory.
set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cache/cpm")
# Prevent cpm_init from trying to tell us where to put cpm.cmake
include(get_cpm)

# Cant use rapids_cpm_init() for now since the `rapids_cpm_download()` creates a
# new scope when importing CPM. Manually do the other commands and import CPM on
# our own with get_cpm
#include("${rapids-cmake-dir}/cpm/detail/load_preset_versions.cmake")
#rapids_cpm_load_preset_versions()

# Print CMake settings when verbose output is enabled
message(VERBOSE "PROJECT_NAME: " ${PROJECT_NAME})
message(VERBOSE "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(VERBOSE "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(VERBOSE "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(VERBOSE "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(VERBOSE "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(VERBOSE "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(VERBOSE "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(VERBOSE "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(VERBOSE "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(VERBOSE "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(VERBOSE "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(VERBOSE "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(VERBOSE "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})
message(VERBOSE "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(VERBOSE "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(VERBOSE "CMAKE_INSTALL_FULL_INCLUDEDIR: " ${CMAKE_INSTALL_FULL_INCLUDEDIR})
message(VERBOSE "CMAKE_INSTALL_FULL_LIBDIR: " ${CMAKE_INSTALL_FULL_LIBDIR})
message(VERBOSE "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(VERBOSE "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(VERBOSE "CMAKE_FIND_ROOT_PATH: " ${CMAKE_FIND_ROOT_PATH})
message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE: " ${CMAKE_LIBRARY_ARCHITECTURE})
message(VERBOSE "FIND_LIBRARY_USE_LIB64_PATHS: " ${FIND_LIBRARY_USE_LIB64_PATHS})

# Load direct physical package dependencies first, so we fail early.
find_package(Protobuf REQUIRED)
find_package(CUDAToolkit REQUIRED) # Required by Morpheus. Fail early if we don't have it.

if(MORPHEUS_BUILD_BENCHMARKS)
  # google benchmark
  # - Expects package to pre-exist in the build environment
  # ================
  rapids_find_package(benchmark REQUIRED
          GLOBAL_TARGETS      benchmark::benchmark
          BUILD_EXPORT_SET    ${PROJECT_NAME}-exports
          INSTALL_EXPORT_SET  ${PROJECT_NAME}-exports
          FIND_ARGS
          CONFIG
          )
endif()

if(MORPHEUS_BUILD_TESTS)
  # google test
  # - Expects package to pre-exist in the build environment
  # ===========
  rapids_find_package(GTest REQUIRED
          GLOBAL_TARGETS      GTest::gtest GTest::gmock GTest::gtest_main GTest::gmock_main
          BUILD_EXPORT_SET    ${PROJECT_NAME}-exports
          INSTALL_EXPORT_SET  ${PROJECT_NAME}-exports
          FIND_ARGS
          CONFIG
          )
endif()

# libcudacxx -- get an explicit lubcudacxx build, matx tries to pull a tag that doesn't exist.
# =========
set(LIBCUDACXX_VERSION "1.8.0" CACHE STRING "Version of libcudacxx to use")
include(deps/Configure_libcudacxx)

# matx
# ====
set(MATX_VERSION "0.1.0" CACHE STRING "Version of MatX to use")
include(deps/Configure_matx)

# pybind11
# =========
set(PYBIND11_VERSION "2.8.1" CACHE STRING "Version of Pybind11 to use")
include(deps/Configure_pybind11)

# RD-Kafka
# =====
set(RDKAFKA_VERSION 1.6.2)
include(deps/Configure_rdkafka)

# SRF (Should come after all third party but before NVIDIA repos)
# =====
set(SRF_VERSION 22.09 CACHE STRING "Which version of SRF to use")
include(deps/Configure_srf)

# CuDF
# =====
set(CUDF_VERSION "${MORPHEUS_RAPIDS_VERSION}" CACHE STRING "Which version of cuDF to use")
include(deps/Configure_cudf)

# Triton-client
# =====
set(TRITONCLIENT_VERSION "${MORPHEUS_RAPIDS_VERSION}" CACHE STRING "Which version of TritonClient to use")
include(deps/Configure_TritonClient)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
