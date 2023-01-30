# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "cuda_utils")

find_package(pybind11 REQUIRED)

# Place the two cuda sources in their own target and disable IWYU for that target.
add_library(cuda_utils_objs
  OBJECT
    ${MORPHEUS_LIB_ROOT}/src/utilities/matx_util.cu
    ${MORPHEUS_LIB_ROOT}/src/utilities/type_util.cu
)

set_target_properties(
    cuda_utils_objs
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      C_INCLUDE_WHAT_YOU_USE ""
      CXX_INCLUDE_WHAT_YOU_USE ""
      EXPORT_COMPILE_COMMANDS OFF
)

target_include_directories(cuda_utils_objs
    PUBLIC
      "${MORPHEUS_LIB_ROOT}/include"
      cudf::cudf
      matx::matx
      mrc::pymrc
)

target_link_libraries(cuda_utils_objs
    PUBLIC
      cudf::cudf
      matx::matx
      mrc::pymrc
)

add_library(cuda_utils
    SHARED
      $<TARGET_OBJECTS:cuda_utils_objs>
      ${MORPHEUS_LIB_ROOT}/src/objects/dev_mem_info.cpp
      ${MORPHEUS_LIB_ROOT}/src/objects/table_info.cpp
      ${MORPHEUS_LIB_ROOT}/src/objects/tensor_object.cpp
      ${MORPHEUS_LIB_ROOT}/src/utilities/tensor_util.cpp
      ${MORPHEUS_LIB_ROOT}/src/utilities/type_util_detail.cpp
)

target_include_directories(cuda_utils
    PUBLIC
      "${MORPHEUS_LIB_ROOT}/include"
)

target_link_libraries(cuda_utils
    PUBLIC
      mrc::pymrc
      matx::matx
      cudf::cudf
      Python3::NumPy
      pybind11::pybind11
)

set_target_properties(
    cuda_utils
    PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
)

set_target_properties(cuda_utils
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_utils
)

message(STATUS " Install dest: (cuda_utils) ${MORPHEUS_LIB_INSTALL_DIR}")
install(
    TARGETS
      cuda_utils
    EXPORT
      ${PROJECT_NAME}-exports
    LIBRARY DESTINATION
      "${MORPHEUS_LIB_INSTALL_DIR}"
    COMPONENT Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(cuda_utils ${MORPHEUS_LIB_ROOT})
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
