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

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

morpheus_utils_initialize_cpm(MORPHEUS_CACHE_DIR)

if (VERBOSE)
  morpheus_utils_print_config()
endif()

# First, load the package_config functions
include(${CMAKE_CURRENT_LIST_DIR}/package_config/register_api.cmake)

# if(MORPHEUS_SUPPORT_DOCA)

#   morpheus_configure_libmd()

#   morpheus_configure_libbsd()

#   set(doca_ROOT "/opt/mellanox/doca")

#   find_package(doca REQUIRED)
# endif()

# Load direct physical package dependencies first, so we fail early. Add all dependencies to our export set
rapids_find_package(Protobuf
  REQUIRED
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

find_package(CUDAToolkit REQUIRED)

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
morpheus_utils_configure_libcudacxx()

# matx
# ====
morpheus_utils_configure_matx()

# pybind11
# =========
morpheus_utils_configure_pybind11()

# RD-Kafka
# =====
morpheus_utils_configure_rdkafka()

# MRC (Should come after all third party but before NVIDIA repos)
# =====
morpheus_utils_configure_mrc()

# CuDF
# =====
morpheus_utils_configure_cudf()

# Triton-client
# =====
morpheus_utils_configure_tritonclient()

# # Finally, install the DOCA components if necessary
# if(MORPHEUS_SUPPORT_DOCA)

#   # libmd
#   # Must come before bsd
#   # =====
#   morpheus_configure_libmd()

#   # libbsd
#   # =====
#   morpheus_configure_libbsd()

#   # gdrcopy
#   # =====
#   morpheus_configure_gdrcopy()

# endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
