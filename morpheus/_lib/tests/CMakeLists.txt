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

list(APPEND CMAKE_MESSAGE_CONTEXT "tests")

# Keep all source files sorted
add_executable(test_libmorpheus
  # test_cuda.cu
  test_main.cpp
  test_masking.cpp
  test_tensor.cpp
  test_type_util_detail.cpp
)

target_link_libraries(test_libmorpheus
  PRIVATE
    cuda_utils
    srf::pysrf
    GTest::gtest
    matx::matx
    pybind11::embed
)

add_test(
  NAME test_libmorpheus
  COMMAND $<TARGET_FILE:test_libmorpheus>
)

set_target_properties(test_libmorpheus
  PROPERTIES
    INSTALL_RPATH "$ORIGIN/.."
)

install(
    TARGETS
      test_libmorpheus
    RUNTIME DESTINATION
      "${MORPHEUS_LIB_INSTALL_DIR}/tests"
    COMPONENT Wheel
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
