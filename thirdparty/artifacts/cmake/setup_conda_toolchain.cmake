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


# Conda / Vcpkg configuration
list(APPEND CMAKE_MESSAGE_CONTEXT "toolchain")
message(STATUS "Configuring Toolchain Components")

rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
  message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
  set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE STRING "" FORCE)
endif()

message(STATUS "Prepending CONDA_PREFIX ($ENV{CONDA_PREFIX}) to CMAKE_FIND_ROOT_PATH")
list(PREPEND CMAKE_FIND_ROOT_PATH "$ENV{CONDA_PREFIX}")
list(REMOVE_DUPLICATES CMAKE_FIND_ROOT_PATH)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
