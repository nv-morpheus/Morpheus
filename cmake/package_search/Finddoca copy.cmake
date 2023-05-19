# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

set(CMAKE_FIND_DEBUG_MODE ON)

# Does not appear to be working
function(my_check validator_result_var item)

  # set(${validator_result_var} FALSE PARENT_SCOPE)

  if(DEFINED ENV{CONDA_PREFIX})
    set(conda_prefix $ENV{CONDA_PREFIX})
    cmake_path(IS_PREFIX conda_prefix "${item}" NORMALIZE is_relative)

    # message(STATUS "Checking $ENV{CONDA_PREFIX} IS_PREFIX ${item}: ${is_relative}")

    if(is_relative)
      set(${validator_result_var} FALSE PARENT_SCOPE)
    endif()
  endif()

  # message(STATUS "  --Checking ${item}: ${${validator_result_var}}")
endfunction()

# Find pkg-config manually. The conda one does weird things and does not work right
find_program(PKG_CONFIG_EXECUTABLE
  NAMES pkg-config
  # HINTS
  #   /usr/local/sbin
  #   /usr/local/bin
  #   /usr/sbin
  #   /usr/bin
  #   /sbin
  #   /bin
  # NO_DEFAULT_PATH
  VALIDATOR my_check
)

set(CMAKE_FIND_DEBUG_MODE OFF)

# set(saved_pkg_config_path $ENV{PKG_CONFIG_PATH})

# if (BASE_PKG_CONFIG_EXE)
#   # Get the default search paths from the host pkg-config
#   execute_process(
#     COMMAND ${BASE_PKG_CONFIG_EXE} --variable pc_path pkg-config
#     OUTPUT_VARIABLE BASE_PKG_CONFIG_PATHS
#   )

#   set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${BASE_PKG_CONFIG_PATHS}}")
# endif()

find_package(PkgConfig QUIET)

# set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO STDOUT)

# Before calling pkg-config, we need to set PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 to enable system paths which are skipped in
# conda builds
set(ENV:{PKG_CONFIG_ALLOW_SYSTEM_CFLAGS} 1)

pkg_check_modules(doca QUIET
  IMPORTED_TARGET
  NO_CMAKE_PATH
  NO_CMAKE_ENVIRONMENT_PATH
  GLOBAL
  doca doca-gpunetio-device
)

unset(ENV:{PKG_CONFIG_ALLOW_SYSTEM_CFLAGS})

# set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO NONE)

# # Now restore the environment variable
# set(ENV{PKG_CONFIG_PATH} ${saved_pkg_config_path})

# unset(saved_pkg_config_path)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(doca
  FOUND_VAR doca_FOUND
  REQUIRED_VARS
    doca_LIBRARIES
    doca_LINK_LIBRARIES
    doca_LIBRARY_DIRS
    doca_LDFLAGS
    doca_LDFLAGS_OTHER
    doca_INCLUDE_DIRS
    doca_CFLAGS
    doca_CFLAGS_OTHER
  VERSION_VAR doca_VERSION
)

mark_as_advanced(
  doca_LIBRARIES
  doca_LINK_LIBRARIES
  doca_LIBRARY_DIRS
  doca_LDFLAGS
  doca_LDFLAGS_OTHER
  doca_INCLUDE_DIRS
  doca_CFLAGS
  doca_CFLAGS_OTHER
)

if (doca_FOUND AND NOT TARGET doca::doca)

  # Add an alias to the imported target
  add_library(doca::doca ALIAS PkgConfig::doca)

  set(name "doca")

  # Now add it to the list of packages to install
  rapids_export_package(INSTALL doca
    ${PROJECT_NAME}-core-exports
    GLOBAL_TARGETS doca::doca
  )

  # Overwrite the default package contents
  configure_file("${CMAKE_SOURCE_DIR}/external/utilities/cmake/morpheus_utils/package_config/hwloc/templates/pkgconfig_package.cmake.in"
    "${CMAKE_BINARY_DIR}/rapids-cmake/${PROJECT_NAME}-core-exports/install/package_${name}.cmake" @ONLY)

  unset(name)

  # morpheus_utils_print_target_properties(
  #   TARGETS
  #     PkgConfig::doca
  #   WRITE_TO_FILE
  # )
endif()
