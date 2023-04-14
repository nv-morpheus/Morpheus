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
list(APPEND CMAKE_MESSAGE_CONTEXT "_lib")

########### morpheus ###########
set(MORPHEUS_LIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

#----------morpheus_utils---------
include(cmake/libraries/morpheus_utils.cmake)

#----------cudf_helpers---------
# had to inline the cudf_helpers.cmake file because of scikit-build changes in https://github.com/scikit-build/scikit-build/pull/871
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


#----------lib_morpheus---------
include(cmake/libraries/morpheus.cmake)

########### py_morpheus ########
# Set the default link targets to avoid repeating this
morpheus_utils_python_package_set_default_link_targets(morpheus mrc::pymrc)

#----------morpheus._lib.stages---------
include(cmake/python_modules/stages.cmake)

#----------morpheus._lib.messages---------
include(cmake/python_modules/messages.cmake)

#----------morpheus._lib.modules---------
include(cmake/python_modules/modules.cmake)

#----------morpheus._lib.common---------
include(cmake/python_modules/common.cmake)

if (MORPHEUS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
