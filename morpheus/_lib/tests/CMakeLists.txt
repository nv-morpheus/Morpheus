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

add_library(
    morpheus_test_utilities
      test_utils/common.cpp
)

target_link_libraries(
  morpheus_test_utilities
  PRIVATE
    GTest::gtest
    morpheus
)

function (add_morpheus_test TEST_NAME TEST_FILES)

  add_executable(
    test_${TEST_NAME}
    ${TEST_FILES}
  )

  target_link_libraries(
    test_${TEST_NAME}
    PRIVATE
      GTest::gtest
      GTest::gtest_main
      matx::matx
      morpheus
      morpheus_test_utilities
      pybind11::embed
  )

  add_test(
    NAME test_${TEST_NAME}
    COMMAND $<TARGET_FILE:test_${TEST_NAME}>
  )

  set_target_properties(test_${TEST_NAME}
    PROPERTIES
      INSTALL_RPATH "$ORIGIN/.."
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
  )

  install(
    TARGETS
      test_${TEST_NAME}
    RUNTIME DESTINATION
      "${MORPHEUS_LIB_INSTALL_DIR}/tests"
    COMPONENT Wheel
  )

endfunction()

add_morpheus_test(cuda
  test_cuda.cu
)

add_morpheus_test(data_loader
  io/test_data_loader.cpp
  io/test_data_loader_registry.cpp
  io/test_loaders.cpp
)

add_morpheus_test(messages
  messages/test_control_message.cpp
  messages/test_dev_doc_ex3.cpp
  messages/test_sliced_message_meta.cpp
)

add_morpheus_test(modules
  modules/test_data_loader_module.cpp
)

add_morpheus_test(deserializers
  test_deserializers.cpp
)

add_morpheus_test(dev_mem_info
  test_dev_mem_info.cpp
)

add_morpheus_test(file_io
  test_file_in_out.cpp
)

add_morpheus_test(matx
  test_matx_util.cpp
)

add_morpheus_test(multi_slices
  test_multi_slices.cpp
)

add_morpheus_test(tensor
  test_tensor.cpp
)

add_morpheus_test(type_util
  test_type_util.cpp
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
