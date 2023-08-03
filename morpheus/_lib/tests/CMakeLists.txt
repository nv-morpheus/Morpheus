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

list(APPEND CMAKE_MESSAGE_CONTEXT "tests")

find_package(pybind11 REQUIRED)


add_executable(test_libmorpheus
  io/test_data_loader_registry.cpp
  io/test_data_loader.cpp
  io/test_loaders.cpp
  messages/test_control_message.cpp
  messages/test_dev_doc_ex3.cpp
  messages/test_sliced_message_meta.cpp
  modules/test_data_loader_module.cpp
  test_cuda.cu
  test_deserializers.cpp
  test_dev_mem_info.cpp
  test_file_in_out.cpp
  test_matx_util.cpp
  test_multi_slices.cpp
  test_tensor.cpp
  test_type_util.cpp
  test_utils/common.cpp
)

target_link_libraries(test_libmorpheus
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    matx::matx
    morpheus
    pybind11::embed
)

add_test(
  NAME test_libmorpheus
  COMMAND $<TARGET_FILE:test_libmorpheus>
)

set_target_properties(test_libmorpheus
  PROPERTIES
    INSTALL_RPATH "$ORIGIN/.."
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

install(
  TARGETS
    test_libmorpheus
  RUNTIME DESTINATION
    "${MORPHEUS_LIB_INSTALL_DIR}/tests"
  COMPONENT Wheel
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
