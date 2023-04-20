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
      morpheus_utils
      Python::Module
      Python::NumPy
    OUTPUT_TARGET
      cudf_helpers_target
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pyarrow; print(pyarrow.get_include())"
  OUTPUT_VARIABLE PYARROW_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(targets_using_arrow_headers ${cudf_helpers_target})
foreach(target IN LISTS targets_using_arrow_headers)
  target_include_directories(${target} PRIVATE "${PYARROW_INCLUDE_DIR}")
endforeach()

# This target generates headers used by other parts of the code base.
# The C++ checks used in CI need these headers but don't require an actual build.
# The `morpheus_style_checks` target allows these to be generated without a full build of Morpheus.
add_dependencies(${PROJECT_NAME}_style_checks ${cudf_helpers_target})

# We don't have control over the C++ code that cython generates, suppress the volatile warning raised by the compiler
target_compile_options(${cudf_helpers_target} PRIVATE -Wno-volatile)

# Disable clang-tidy and IWYU for cython generated code
set_target_properties(
  ${cudf_helpers_target}
    PROPERTIES
      CXX_CLANG_TIDY ""
      C_INCLUDE_WHAT_YOU_USE ""
      CXX_INCLUDE_WHAT_YOU_USE ""
      EXPORT_COMPILE_COMMANDS OFF
)
