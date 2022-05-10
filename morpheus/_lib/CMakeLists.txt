# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

find_package(pybind11 REQUIRED)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  message(STATUS "Performing inplace build for Python modules and library dependencies. Use `pip install -e .`\
   from the root folder to install Morpheus.")
endif()

########### morpheus ###########
set(MORPHEUS_LIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

file(RELATIVE_PATH install_relative ${MORPHEUS_PY_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
set(MORPHEUS_LIB_INSTALL_DIR "${MORPHEUS_PY_INSTALL_DIR}/${install_relative}")
unset(install_relative)

#----------cuda_utils---------
include(cmake/libraries/cuda_utils.cmake)

#----------cudf_helpers---------
include(cmake/libraries/cudf_helpers.cmake)

#----------lib_morpheus---------
include(cmake/libraries/morpheus.cmake)

########### py_morpheus ########
# Create list to store all pybind11 targets
set(PYBIND11_TARGETS "")

#----------morpheus._lib.stages---------
include(cmake/python_modules/stages.cmake)

#----------morpheus._lib.messages---------
include(cmake/python_modules/messages.cmake)

#----------morpheus._lib.common---------
include(cmake/python_modules/common.cmake)

#----------morpheus._lib.file_types---------
include(cmake/python_modules/file_types.cmake)

# message(STATUS "PYBIND11_TARGETS: ${PYBIND11_TARGETS}")

# # Build up the list of output files
# set(stub_output_files "")

# foreach(pybind11_target ${PYBIND11_TARGETS})

#   # Determine the stub file in the build folder
#   string(REPLACE "." "/" pybind11_target_path ${pybind11_target})

#   message(STATUS "pybind11_target_path: ${pybind11_target_path}")

#   set(build_stub_file "${PROJECT_BINARY_DIR}/${pybind11_target_path}/__init__.pyi")

#   message(STATUS "build_stub_file: ${build_stub_file}")
#   message(STATUS "Stub gen command: ${Python3_EXECUTABLE} -m pybind11_stubgen ${pybind11_target} --no-setup-py -o ${pybind11_target_path} --root-module-suffix \"\"")

#   # Create the custom command to generate the stub
#   add_custom_command(
#     OUTPUT  ${build_stub_file}
#     COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen ${pybind11_target} --no-setup-py -o ./ --root-module-suffix \"\"
#     DEPENDS ${pybind11_target} copy_python_source
#     COMMENT "Building stub for python module ${pybind11_target}..."
#     WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
#   )

#   # Add the build stub to the outputs
#   list(APPEND stub_output_files ${build_stub_file})

#   if(MORPHEUS_PYTHON_INPLACE_BUILD)
#     # Determine the src stub path
#     set(src_stub_file "${PROJECT_SOURCE_DIR}/${pybind11_target_path}/__init__.pyi")

#     # Copy from build back to src
#     add_custom_command(
#       OUTPUT  ${src_stub_file}
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different ${build_stub_file} ${src_stub_file}
#       DEPENDS ${build_stub_file}
#       COMMENT "Copying stub ${build_stub_file} to ${src_stub_file}"
#     )

#     # Add the src stub to the outputs
#     list(APPEND stub_output_files ${src_stub_file})
#   endif()

# endforeach()

# message(STATUS "stub_output_files: ${stub_output_files}")

# # Add a custom target to depend on the custom command
# add_custom_target(build_pybind11_stubs ALL
#   DEPENDS ${stub_output_files}
# )


list(POP_BACK CMAKE_MESSAGE_CONTEXT)
