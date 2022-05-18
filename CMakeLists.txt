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


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)


list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus")

# Global options
option(BUILD_SHARED_LIBS "Default value for whether or not to build shared or static libraries" ON)
option(MORPHEUS_BUILD_BENCHMARKS "Whether or not to build benchmarks" OFF)
option(MORPHEUS_BUILD_EXAMPLES "Whether or not to build examples" OFF)
option(MORPHEUS_BUILD_TESTS "Whether or not to build tests" OFF)
option(MORPHEUS_USE_CCACHE "Enable caching compilation results with ccache" OFF)
option(MORPHEUS_USE_CLANG_TIDY "Enable running clang-tidy as part of the build process" OFF)
option(MORPHEUS_USE_CONDA "Enables finding dependencies via conda instead of vcpkg.
  Note: This will disable vcpkg. All dependencies must be installed first in the conda environment" OFF)
set(MORPHEUS_PY_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/wheel" CACHE STRING "Location to install the python directory")

set(MORPHEUS_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE PATH "Directory to contain all CPM and CCache data")
mark_as_advanced(MORPHEUS_CACHE_DIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
SET(CMAKE_INSTALL_RPATH "$ORIGIN")

enable_testing()

# CMake path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Add the RAPIDS cmake helper scripts
include(import-rapids-cmake)

# Configure CUDA architecture
include(configure_cuda_architecture)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Toolchain selection (Conda/VCPKG/custom) and configuration
include(cmake/configure_toolchain.cmake)

# Project definition
project(morpheus LANGUAGES C CXX CUDA)

# Ccache configuration
include(setup_cache)

# Configure all dependencies
include(cmake/dependencies.cmake)

####################################
# - Post dependencies setup --------
include(cmake/setup_compiler.cmake)

##################################
##### Morpheus Python Setup ######
##################################


# Include the main trtlab code
add_subdirectory(morpheus)

# Get all of the python files to copy to the build directory
file(GLOB_RECURSE MORPHEUS_PYTHON_FILES
  RELATIVE ${PROJECT_SOURCE_DIR}
  LIST_DIRECTORIES FALSE
  CONFIGURE_DEPENDS
  "morpheus/*.py"
  "morpheus/py.typed"
)

set(MORPHEUS_ROOT_PYTHON_FILES
  "pyproject.toml"
  "setup.cfg"
  "setup.py"
  "versioneer.py"
  "MANIFEST.in"
)

# Add the root python files to the list
list(PREPEND MORPHEUS_PYTHON_FILES ${MORPHEUS_ROOT_PYTHON_FILES})

set(OUTPUT_PYTHON_FILES "")

# For each file, copy if its different
foreach(py_file ${MORPHEUS_PYTHON_FILES})

  add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/${py_file}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROJECT_SOURCE_DIR}/${py_file} ${PROJECT_BINARY_DIR}/${py_file}
    DEPENDS ${PROJECT_SOURCE_DIR}/${py_file}
    COMMENT "Copying ${py_file} to ${PROJECT_BINARY_DIR}/${py_file}"
  )

  list(APPEND OUTPUT_PYTHON_FILES ${PROJECT_BINARY_DIR}/${py_file})

endforeach()

add_custom_target(copy_python_source ALL
  DEPENDS ${OUTPUT_PYTHON_FILES}
)

# Manually install the below files. install(DIRECTORY) doesnt work well and
# makes it impossible to get these files and MORPHEUS_PYTHON_FILES in one command.
install(
  FILES ${MORPHEUS_ROOT_PYTHON_FILES}
  DESTINATION ${MORPHEUS_PY_INSTALL_DIR}
  COMPONENT Wheel
)

# Use the install(DIRECTORY) to keep the files tructure
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/morpheus
  DESTINATION ${MORPHEUS_PY_INSTALL_DIR}
  COMPONENT Wheel
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "py.typed"
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/morpheus/data
  DESTINATION ${MORPHEUS_PY_INSTALL_DIR}/morpheus
  COMPONENT Wheel
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
