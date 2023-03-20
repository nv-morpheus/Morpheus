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

morpheus_add_cython_library(
    cudf_helpers
    PYX_FILE
      "${MORPHEUS_LIB_ROOT}/cudf_helpers.pyx"
    LINK_TARGETS
      cuda_utils
    OUTPUT_TARGET
      cudf_helpers_target
)

# This target generates headers used by other parts of the code base.
# The C++ checks used in CI need these headers but don't require an actual build.
# The `morpheus_style_checks` target allows these to be generated without a full build of Morpheus.
add_dependencies(${PROJECT_NAME}_style_checks ${cudf_helpers_target})

# Disable clang-tidy and IWYU for cython generated code
set_target_properties(
  ${cudf_helpers_target}
    PROPERTIES
      CXX_CLANG_TIDY ""
      C_INCLUDE_WHAT_YOU_USE ""
      CXX_INCLUDE_WHAT_YOU_USE ""
      EXPORT_COMPILE_COMMANDS OFF
)
