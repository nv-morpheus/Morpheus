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

# Must create a custom target to hold all python dependencies before first module call
add_custom_target(all_python_targets ALL
  DEPENDS copy_python_source # depend on copying the source so the full python package is built with this target
)

#----------cuda_utils---------
include(cmake/libraries/cuda_utils.cmake)

#----------cudf_helpers---------
include(cmake/libraries/cudf_helpers.cmake)

#----------lib_morpheus---------
include(cmake/libraries/morpheus.cmake)

########### py_morpheus ########
#----------morpheus._lib.stages---------
include(cmake/python_modules/stages.cmake)

#----------morpheus._lib.messages---------
include(cmake/python_modules/messages.cmake)

#----------morpheus._lib.common---------
include(cmake/python_modules/common.cmake)

#----------morpheus._lib.file_types---------
include(cmake/python_modules/file_types.cmake)

function(print_target_properties)
  set(prefix F_ARGV)
  set(options WRITE_TO_FILE PRINT_NOTFOUND)
  set(singleValueArgs WRITE_TO_FILE_NAME)
  set(multiValueArgs TARGETS PROPERTIES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${options}"
      "${singleValueArgs}"
      "${multiValueArgs}"
      ${ARGN})

  list(SORT F_ARGV_PROPERTIES)
  list(REMOVE_DUPLICATES F_ARGV_PROPERTIES)
  foreach(CURRENT_TARGET ${F_ARGV_TARGETS})
    set(current_properties_list "")

    message(STATUS "[Target Properties '${CURRENT_TARGET}']")
    message(STATUS "=====================================================")
    foreach(CURRENT_PROPERTY ${F_ARGV_PROPERTIES})
      string(STRIP ${CURRENT_PROPERTY} CURRENT_PROPERTY) # Remove trailing/leading whitespace
      string(REGEX MATCHALL "^[^#].*$" CURRENT_PROPERTY ${CURRENT_PROPERTY}) # Comments hack

      if (NOT CURRENT_PROPERTY)
        continue()
      endif()

      get_target_property(CURRENT_PROPERTY_VALUE ${CURRENT_TARGET} ${CURRENT_PROPERTY})
      if((NOT F_ARGV_PRINT_NOTFOUND) AND ("${CURRENT_PROPERTY_VALUE}" STREQUAL "CURRENT_PROPERTY_VALUE-NOTFOUND"))
      else()
        if ("${CURRENT_PROPERTY_VALUE}" STREQUAL "")
          set(CURRENT_PROPERTY_VALUE "<EMPTY_STR>")
        else()
          set(CURRENT_PROPERTY_VALUE "${CURRENT_PROPERTY_VALUE}")
        endif()

        string(JOIN ", " CURRENT_PROPERTY_VALUE ${CURRENT_PROPERTY_VALUE})
        list(APPEND current_properties_list "${CURRENT_PROPERTY} -> \"${CURRENT_PROPERTY_VALUE}\"")
      endif()
    endforeach()

    foreach(PROPERTY_PAIR ${current_properties_list})
      if (F_ARGV_WRITE_TO_FILE)
        FILE(APPEND "${CMAKE_BINARY_DIR}/morpheus_utils_debug_target_properties.txt" ${PROPERTY_PAIR})
      endif()
      message(STATUS "${PROPERTY_PAIR}")
    endforeach()
  endforeach()
endfunction()

file(STRINGS "${CMAKE_SOURCE_DIR}/test.cmake.props" PROPS_TO_GET)
print_target_properties(
    TARGETS
      cudf::cudf
      matx::matx
      mrc::pymrc
      cuda_utils_objs
      cuda_utils
    PROPERTIES
      ${PROPS_TO_GET}
    )

if (MORPHEUS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
