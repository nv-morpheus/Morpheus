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

list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_utils")

# find_package(pybind11 REQUIRED)


add_library(morpheus_utils
  SHARED
    ${MORPHEUS_LIB_ROOT}/src/objects/table_info_data.cpp
)

target_include_directories(morpheus_utils
  PUBLIC
    "${MORPHEUS_LIB_ROOT}/include"
)

target_link_libraries(morpheus_utils
  PUBLIC
    cudf::cudf
  PRIVATE
    glog::glog
)

# set_target_properties(
#   morpheus_utils
#   PROPERTIES
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
# )

install(
  TARGETS
  morpheus_utils
  EXPORT
    ${PROJECT_NAME}-exports
  COMPONENT Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus_utils ${MORPHEUS_LIB_ROOT})
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
