# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "simple_cpp_stage")

set(Python3_FIND_VIRTUALENV "FIRST")
set(Python3_FIND_STRATEGY "LOCATION")
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)

pybind11_add_module(morpheus_example MODULE "${CMAKE_CURRENT_SOURCE_DIR}/_lib/pass_thru.cpp")

add_library(${PROJECT_NAME}::morpheus_example ALIAS morpheus_example)

target_link_libraries(morpheus_example
    PUBLIC
      morpheus
      srf::pysrf
)

target_include_directories(morpheus_example
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${MORPHEUS_ROOT}/morpheus/_lib/include>
)

set_target_properties(morpheus_example PROPERTIES CXX_VISIBILITY_PRESET hidden)
inplace_build_copy(morpheus_example "${CMAKE_CURRENT_SOURCE_DIR}/_lib")

message(STATUS " Install dest: (morpheus_example) ${CMAKE_CURRENT_BINARY_DIR}/_lib")
install(
    TARGETS
    morpheus_example
    LIBRARY DESTINATION
      "${CMAKE_CURRENT_BINARY_DIR}/_lib"
)


list(POP_BACK CMAKE_MESSAGE_CONTEXT)
