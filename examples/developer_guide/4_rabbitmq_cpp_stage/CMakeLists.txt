# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "4_rabbitmq_cpp_stage")

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# Set the cache to be the same to allow for CCache to be used effectively
set(MORPHEUS_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MORPHEUS_CACHE_DIR)

# Add the Conda environment to the prefix path and add the CMake files
list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../external/utilities/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(morpheus_utils/load)

project(4_rabbitmq_cpp_stage
  VERSION 24.03.00
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)

morpheus_utils_initialize_cpm(MORPHEUS_CACHE_DIR)

# Ensure CPM is initialized
rapids_cpm_init()

# Set the option prefix to match the outer project before including. Must be before find_package(morpheus)
set(OPTION_PREFIX "MORPHEUS")

morpheus_utils_python_configure()
include(cmake/dependencies.cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

morpheus_utils_create_python_package(morpheus_rabbit
  PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

add_subdirectory(src/rabbitmq_cpp_stage/_lib)

# Complete the python package
if(MORPHEUS_PYTHON_INPLACE_BUILD)
  list(APPEND extra_args "IS_INPLACE")
endif()

if(TARGET morpheus-package-install)
  list(APPEND extra_args "PYTHON_DEPENDENCIES" "morpheus-package-install")
endif()

morpheus_utils_build_python_package(morpheus_rabbit ${extra_args})

# -----------
# pybind11_add_module(morpheus_rabbit MODULE "${CMAKE_CURRENT_SOURCE_DIR}/_lib/rabbitmq_source.cpp")

# add_library(${PROJECT_NAME}::morpheus_rabbit ALIAS morpheus_rabbit)

# target_link_libraries(morpheus_rabbit
# PUBLIC
# morpheus
# rabbitmq
# SimpleAmqpClient
# mrc::pymrc
# )

# target_include_directories(morpheus_rabbit
# PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
# $<BUILD_INTERFACE:${MORPHEUS_ROOT}/morpheus/_lib/include>
# rabbitmq
# ${SimpleAmqpClient_SOURCE_DIR}/src
# )

# set_target_properties(morpheus_rabbit PROPERTIES CXX_VISIBILITY_PRESET hidden)

# if(MORPHEUS_PYTHON_INPLACE_BUILD)
# set_target_properties(morpheus_rabbit
# PROPERTIES
# INSTALL_RPATH "${PROJECT_SOURCE_DIR}/morpheus/_lib"
# )
# else()
# # TODO: https://github.com/nv-morpheus/Morpheus/issues/331
# set_target_properties(morpheus_rabbit
# PROPERTIES
# INSTALL_RPATH "${Python3_SITELIB}/morpheus/_lib"
# )
# endif()

# morpheus_utils_inplace_build_copy(morpheus_rabbit "${CMAKE_CURRENT_SOURCE_DIR}/_lib")

# message(STATUS " Install dest: (morpheus_rabbit) ${CMAKE_CURRENT_BINARY_DIR}/_lib")
# install(
# TARGETS
# morpheus_rabbit
# LIBRARY DESTINATION
# "${CMAKE_CURRENT_BINARY_DIR}/_lib"
# )
list(POP_BACK CMAKE_MESSAGE_CONTEXT)
