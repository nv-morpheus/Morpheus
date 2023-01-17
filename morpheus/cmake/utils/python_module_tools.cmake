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

## TODO: these need to be extracted to a cmake utilities repo

function(inplace_build_copy TARGET_NAME INPLACE_DIR)
  message(STATUS " Inplace build: (${TARGET_NAME}) ${INPLACE_DIR}")

  add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${INPLACE_DIR}
    COMMENT "Moving target ${TARGET_NAME} to ${INPLACE_DIR} for inplace build"
  )

  # See if there are any resources associated with this target
  get_target_property(target_resources ${TARGET_NAME} RESOURCE)

  if(target_resources)

    # Get the build and src locations
    get_target_property(target_build_dir ${TARGET_NAME} BINARY_DIR)

    set(resource_outputs "")

    # Create the copy command for each resource
    foreach(resource ${target_resources})
      # Get the relative path to the build directory
      file(RELATIVE_PATH relative_resource ${target_build_dir} ${resource})

      add_custom_command(
        OUTPUT  ${INPLACE_DIR}/${relative_resource}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${resource} ${INPLACE_DIR}/${relative_resource}
        DEPENDS ${resource}
        COMMENT "Copying stub ${resource} to ${INPLACE_DIR}/${relative_resource}"
      )
      list(APPEND resource_outputs ${INPLACE_DIR}/${relative_resource})
    endforeach()

    # Final target to depend on the copied files
    add_custom_target(${TARGET_NAME}-resource-inplace ALL
      DEPENDS ${resource_outputs}
    )
  endif()

endfunction()

#[=======================================================================[
@brief : given a module name, and potentially a root path, resolves the
fully qualified python module path. If MODULE_ROOT is not provided, it
will default to ${CMAKE_CURRENT_SOURCE_DIR} -- the context of
the caller.

ex. resolve_python_module_name(my_module MODULE_ROOT morpheus/_lib)
results --
  MODULE_TARGET_NAME:   morpheus._lib.my_module
  OUTPUT_MODULE_NAME:   my_module
  OUTPUT_RELATIVE_PATH: morpheus/_lib

resolve_python_module_name <MODULE_NAME>
                           [MODULE_ROOT]
                           [OUTPUT_TARGET_NAME]
                           [OUTPUT_MODULE_NAME]
                           [OUTPUT_RELATIVE_PATH]
#]=======================================================================]

function(resolve_python_module_name MODULE_NAME)
  set(prefix PYMOD) # Prefix parsed args
  set(flags "")
  set(singleValues
      MODULE_ROOT
      OUTPUT_TARGET_NAME
      OUTPUT_MODULE_NAME
      OUTPUT_RELATIVE_PATH)
  set(multiValues "")

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${flags}"
      "${singleValues}"
      "${multiValues}"
      ${ARGN})

  set(py_module_name ${MODULE_NAME})

  if($MODULE_ROOT)
    set(py_module_path ${MODULE_ROOT})
  else()
    file(RELATIVE_PATH py_rel_path ${MORPHEUS_PY_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
    set(py_module_path ${py_rel_path})
  endif()

  # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
  string(REPLACE "/" "." py_module_namespace ${py_module_path})

  if (PYMOD_OUTPUT_TARGET_NAME)
    set(${PYMOD_OUTPUT_TARGET_NAME} "${py_module_namespace}.${py_module_name}" PARENT_SCOPE)
  endif()
  if (PYMOD_OUTPUT_MODULE_NAME)
    set(${PYMOD_OUTPUT_MODULE_NAME} "${py_module_name}" PARENT_SCOPE)
  endif()
  if (PYMOD_OUTPUT_RELATIVE_PATH)
    set(${PYMOD_OUTPUT_RELATIVE_PATH} "${py_module_path}" PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[
@brief : TODO
ex. add_python_module
results --

add_python_module

#]=======================================================================]
macro(add_python_module MODULE_NAME)
  set(prefix PYMOD)
  set(flags IS_PYBIND11 IS_CYTHON)
  set(singleValues INSTALL_DEST OUTPUT_TARGET MODULE_ROOT PYX_FILE)
  set(multiValues INCLUDE_DIRS LINK_TARGETS SOURCE_FILES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${flags}"
      "${singleValues}"
      "${multiValues}"
      ${ARGN})

  resolve_python_module_name(${MODULE_NAME}
      OUTPUT_TARGET_NAME TARGET_NAME
      OUTPUT_MODULE_NAME MODULE_NAME
      OUTPUT_RELATIVE_PATH SOURCE_RELATIVE_PATH
      )

  # Ensure the custom target all_python_targets has been created
  if (NOT TARGET all_python_targets)
    message(FATAL_ERROR "You must call `add_custom_target(all_python_targets)` before the first call to add_python_module")
  endif()

  # Create the module target
  if (PYMOD_IS_PYBIND11)
    message(STATUS "Adding Pybind11 Module: ${TARGET_NAME}")
    pybind11_add_module(${TARGET_NAME} MODULE ${PYMOD_SOURCE_FILES})
  elseif(PYMOD_IS_CYTHON)
    message(STATUS "Adding Cython Module: ${TARGET_NAME}")
    add_cython_target(${MODULE_NAME} "${PYMOD_PYX_FILE}" CXX PY3)
    add_library(${TARGET_NAME} SHARED ${${MODULE_NAME}} ${PYMOD_SOURCE_FILES})

    # Need to set -fvisibility=hidden for cython according to https://pybind11.readthedocs.io/en/stable/faq.html
    # set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  else()
    message(FATAL_ERROR "Must specify either IS_PYBIND11 or IS_CYTHON")
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")

  set(pymod_link_libs "")
  if (PYMOD_LINK_TARGETS)
    foreach(target IN LISTS PYMOD_LINK_TARGETS)
      list(APPEND pymod_link_libs ${target})
    endforeach()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${pymod_link_libs}
  )

  # Tell CMake to use relative paths in the build directory. This is necessary for relocatable packages
  set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN")

  if (PYMOD_INCLUDE_DIRS)
    target_include_directories(${TARGET_NAME}
        PRIVATE
          "${PYMOD_INCLUDE_DIRS}"
    )
  endif()

  # Cython targets need the current dir for generated files
  if(PYMOD_IS_CYTHON)
    target_include_directories(${TARGET_NAME}
        PUBLIC
          "${CMAKE_CURRENT_BINARY_DIR}"
    )
  endif()

  # Set all_python_targets to depend on this module. This ensures that all python targets have been built before any
  # post build actions are taken. This is often necessary to allow post build actions that load the python modules to
  # succeed
  add_dependencies(all_python_targets ${TARGET_NAME})

  if (MORPHEUS_BUILD_PYTHON_STUBS)
    # Before installing, create the custom command to generate the stubs
    set(pybind11_stub_file "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}/__init__.pyi")

    add_custom_command(
      OUTPUT  ${pybind11_stub_file}
      COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen ${TARGET_NAME} --no-setup-py --log-level WARN -o ./ --root-module-suffix \"\"
      DEPENDS ${TARGET_NAME} all_python_targets
      COMMENT "Building stub for python module ${TARGET_NAME}..."
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )

    # Add a custom target to ensure the stub generation runs
    add_custom_target(${TARGET_NAME}-stubs ALL
      DEPENDS ${pybind11_stub_file}
    )

    # Save the output as a target property
    set_target_properties(${TARGET_NAME} PROPERTIES RESOURCE "${pybind11_stub_file}")

    unset(pybind11_stub_file)
  endif()

  if (PYMOD_INSTALL_DEST)
    message(STATUS " Install dest: (${TARGET_NAME}) ${PYMOD_INSTALL_DEST}")
    install(
      TARGETS
        ${TARGET_NAME}
      EXPORT
        ${PROJECT_NAME}-exports
      LIBRARY
        DESTINATION
          "${PYMOD_INSTALL_DEST}"
        COMPONENT Wheel
      RESOURCE
        DESTINATION
          "${PYMOD_INSTALL_DEST}/${MODULE_NAME}"
        COMPONENT Wheel
    )
  endif()

  # Set the output target
  if (PYMOD_OUTPUT_TARGET)
    set(${PYMOD_OUTPUT_TARGET} "${TARGET_NAME}" PARENT_SCOPE)
  endif()

endmacro()

#[=======================================================================[
@brief : TODO
ex. add_cython_library
results --

add_cython_library

#]=======================================================================]
function(morpheus_add_cython_libraries MODULE_NAME)

  add_python_module(${MODULE_NAME} IS_CYTHON ${ARGN})

endfunction()

#[=======================================================================[
@brief : TODO
ex. add_cython_library
results --

add_cython_library

#]=======================================================================]
function(morpheus_add_pybind11_module MODULE_NAME)

  add_python_module(${MODULE_NAME} IS_PYBIND11 ${ARGN})

endfunction()
