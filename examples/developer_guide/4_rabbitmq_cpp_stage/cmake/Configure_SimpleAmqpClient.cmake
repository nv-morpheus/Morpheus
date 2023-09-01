#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_SimpleAmqpClient version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "SimpleAmqpClient")

  find_package(rabbitmq REQUIRED)

  rapids_cpm_find(SimpleAmqpClient ${version}
    GLOBAL_TARGETS
      SimpleAmqpClient
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  https://github.com/alanxz/SimpleAmqpClient
      GIT_TAG         "v${version}"
      GIT_SHALLOW     TRUE
      OPTIONS         "Rabbitmqc_INCLUDE_DIR ${rabbitmq_SOURCE_DIR}/include"
                      "Rabbitmqc_LIBRARY ${rabbitmq_BINARY_DIR}/librabbitmq/librabbitmq.so"
                      "BUILD_API_DOCS OFF"
                      "BUILD_SHARED_LIBS OFF"
  )

  # Needed to pick up the generated export.h
  target_include_directories(SimpleAmqpClient PUBLIC "${rabbitmq_BINARY_DIR}/include")

  # Suppress #warning deprecation messages from rabbitmq
  target_compile_options(SimpleAmqpClient PRIVATE -Wno-cpp)

endfunction()

find_and_configure_SimpleAmqpClient(${SIMPLE_AMQP_CLIENT_VERSION})
