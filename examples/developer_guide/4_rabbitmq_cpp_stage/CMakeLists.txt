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

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_rabbit")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(Python3_FIND_VIRTUALENV "FIRST")
set(Python3_FIND_STRATEGY "LOCATION")
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)

include(cmake/dependencies.cmake)

pybind11_add_module(morpheus_rabbit MODULE "${CMAKE_CURRENT_SOURCE_DIR}/_lib/rabbitmq_source.cpp")

add_library(${PROJECT_NAME}::morpheus_rabbit ALIAS morpheus_rabbit)

target_link_libraries(morpheus_rabbit
    PUBLIC
      morpheus
      rabbitmq
      SimpleAmqpClient
      srf::pysrf
)

target_include_directories(morpheus_rabbit
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<BUILD_INTERFACE:${MORPHEUS_ROOT}/morpheus/_lib/include>
      rabbitmq
      ${SimpleAmqpClient_SOURCE_DIR}/src
)

set_target_properties(morpheus_rabbit PROPERTIES CXX_VISIBILITY_PRESET hidden)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  set_target_properties(morpheus_rabbit
    PROPERTIES
      INSTALL_RPATH "${PROJECT_SOURCE_DIR}/morpheus/_lib"
  )
else()
  # TODO: https://github.com/nv-morpheus/Morpheus/issues/331
  set_target_properties(morpheus_rabbit
    PROPERTIES
      INSTALL_RPATH "${Python3_SITELIB}/morpheus/_lib"
  )
endif()


inplace_build_copy(morpheus_rabbit "${CMAKE_CURRENT_SOURCE_DIR}/_lib")

message(STATUS " Install dest: (morpheus_rabbit) ${CMAKE_CURRENT_BINARY_DIR}/_lib")
install(
    TARGETS
    morpheus_rabbit
    LIBRARY DESTINATION
      "${CMAKE_CURRENT_BINARY_DIR}/_lib"
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
