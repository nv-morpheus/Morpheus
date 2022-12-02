# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

include_guard()
include(FetchContent)

# Fetch morpheus utilities -- don't use CPM, RAPIDS_CPM, or other external libraries here so
#   we are only relying on CMake to get our core utilities.
function(find_and_configure_morpheus_utils version)
  list(APPEND CMAKE_MESSAGE_CONTEXT "morpheus_utils")

  set(fetchcontent_tmp "${FETCHCONTENT_BASE_DIR}")
  set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cache") # default location
  if (${CPM_SOURCE_CACHE})
    set(FETCHCONTENT_BASE_DIR "${CPM_SOURCE_CACHE}")
  endif()

  FetchContent_Declare(
      morpheus_utils
      # TODO(Devin): Change to https once utilities is public.
      GIT_REPOSITORY /home/drobison/Development/devin-morpheus-utils-public
      GIT_TAG v${version}
      GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(morpheus_utils)
  set(FETCHCONTENT_BASE_DIR "${fetchcontent_tmp}")

  set(MORPHEUS_UTILS_HOME "${morpheus_utils_SOURCE_DIR}" CACHE INTERNAL "Morpheus utils home")
  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endfunction()

find_and_configure_morpheus_utils(${MORPHEUS_UTILS_VERSION})

list(APPEND CMAKE_MODULE_PATH "${MORPHEUS_UTILS_HOME}/cmake")
include(morpheus_utils/load)
