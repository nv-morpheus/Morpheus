#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# ######################################################################################################################
# * CMake properties ------------------------------------------------------------------------------

list(APPEND CMAKE_MESSAGE_CONTEXT "cache")

function(configure_ccache cache_dir_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "ccache")

  find_program(CCACHE_PROGRAM_PATH ccache DOC "Location of ccache executable")

  if (NOT CCACHE_PROGRAM_PATH)
    message(WARNING "CCache option, ${cache_dir_name}, is enabled but ccache was not found. Check ccache installation.")
    return()
  endif()


  message(STATUS "Using ccache: ${CCACHE_PROGRAM_PATH}")

  set(LOCAL_MODULES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set(CCACHE_DIR "${${cache_dir_name}}/ccache")

  message(STATUS "Using ccache directory: ${CCACHE_DIR}")

  # Write or update the ccache configuration file
  configure_file("${LOCAL_MODULES_PATH}/ccache.conf.in" "${CCACHE_DIR}/ccache.conf")

  # Set the ccache options we need
  set(CCACHE_CONFIGPATH "${CCACHE_DIR}/ccache.conf")

  # Because CMake doesn't allow settings variables `CCACHE_COMPILERTYPE=gcc
  # ccache` in CMAKE_C_COMPILER_LAUNCHER, we need to put everything into a
  # single script and use that for CMAKE_C_COMPILER_LAUNCHER. Also, since
  # gxx_linux-64 sets the compiler to c++ instead of g++, we need to set the
  # value of CCACHE_COMPILERTYPE otherwise caching doesn't work correctly. So
  # we need to make separate runners for each language with specific ccache
  # settings for each
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CCACHE_REMOVE_ARGS "^--driver-mode=.*")
  endif()

  # Set the base dir for relative ccache
  set(CCACHE_BASEDIR "${PROJECT_SOURCE_DIR}")

  # Configure ccache for C
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CCACHE_COMPILERTYPE "gcc")
  elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CCACHE_COMPILERTYPE "clang")
  else()
    set(CCACHE_COMPILERTYPE "auto")
  endif()

  configure_file("${LOCAL_MODULES_PATH}/run_ccache.sh.in" "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_c.sh" @ONLY)

  # Configure ccache for CXX
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CCACHE_COMPILERTYPE "gcc")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CCACHE_COMPILERTYPE "clang")
  else()
    set(CCACHE_COMPILERTYPE "auto")
  endif()

  configure_file("${LOCAL_MODULES_PATH}/run_ccache.sh.in" "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_cxx.sh" @ONLY)

  # Configure ccache for CUDA
  set(CCACHE_COMPILERTYPE "nvcc")
  configure_file("${LOCAL_MODULES_PATH}/run_ccache.sh.in" "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_cuda.sh" @ONLY)

  # Finally, set the compiler option
  set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_c.sh" PARENT_SCOPE)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_cxx.sh" PARENT_SCOPE)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/run_ccache_cuda.sh" PARENT_SCOPE)

  # PARENT_SCOPE here so others can use this value
  set(CCACHE_DIR "${CCACHE_DIR}" PARENT_SCOPE)

endfunction()

function(configure_cpm cache_dir_name)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cpm")

  # Set the CPM cache variable
  set(ENV{CPM_SOURCE_CACHE} "${${cache_dir_name}}/cpm")

  message(STATUS "Using CPM source cache: $ENV{CPM_SOURCE_CACHE}")

  # # Set the FetchContent default download folder to be the same as CPM
  # set(FETCHCONTENT_BASE_DIR "${${cache_dir_name}}/fetch" CACHE STRING "" FORCE)

endfunction()

function(check_cache_path cache_dir_name)
  # First, ensure that the current cache dir can be found by find_package/find_path/etc
  if((NOT "${CMAKE_FIND_ROOT_PATH}" STREQUAL "") AND ("${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE}" STREQUAL "ONLY"))

    set(is_contained FALSE)

    # Now check if ${cache_dir_name} is under anything in CMAKE_FIND_ROOT_PATH
    foreach(path_to_search ${CMAKE_FIND_ROOT_PATH})
      # Check if we are contained by the find path
      cmake_path(IS_PREFIX path_to_search ${${cache_dir_name}} is_relative)

      if(is_relative)
        set(is_contained TRUE)
        break()
      endif()
    endforeach()

    if(NOT is_contained)
      message(WARNING "The value for ${cache_dir_name} (${${cache_dir_name}}) is not contained in any CMAKE_FIND_ROOT_PATH (${CMAKE_FIND_ROOT_PATH}). "
                      "This will result in cmake being unable to find any downloaded packages. The cache path has been appended to the back "
                      "of CMAKE_FIND_ROOT_PATH")

      list(APPEND CMAKE_FIND_ROOT_PATH ${${cache_dir_name}})

      set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} PARENT_SCOPE)
    endif()
  endif()
endfunction()

check_cache_path(MORPHEUS_CACHE_DIR)

# Configure CCache if requested
if(MORPHEUS_USE_CCACHE)
  configure_ccache(MORPHEUS_CACHE_DIR)
endif()

configure_cpm(MORPHEUS_CACHE_DIR)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
