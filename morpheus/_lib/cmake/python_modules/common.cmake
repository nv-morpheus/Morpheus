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

morpheus_utils_add_pybind11_module(
    common
    MODULE_ROOT
      "${MORPHEUS_LIB_ROOT}"
    SOURCE_FILES
      "${MORPHEUS_LIB_ROOT}/src/python_modules/common.cpp"
    INCLUDE_DIRS
      "${MORPHEUS_LIB_ROOT}/include"
    LINK_TARGETS
      morpheus
      mrc::pymrc
    OUTPUT_TARGET
      common_target
    INSTALL_DEST
      ${MORPHEUS_LIB_INSTALL_DIR}
)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(${common_target} ${MORPHEUS_LIB_ROOT})
endif()
