# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LIST(APPEND CMAKE_MESSAGE_CONTEXT "gdrcopy")

include_guard()
include(ExternalProject)

function(find_and_configure_gdrcopy version)
  # Check if gdrcopy is available -- download if not
  rapids_cpm_find(gdrcopy ${version}
    GLOBAL_TARGETS
      gdrcopy
    CPM_ARGS
      GIT_REPOSITORY          https://github.com/NVIDIA/gdrcopy.git
      GIT_TAG                 v${version}
      DOWNLOAD_ONLY           TRUE
  )

  if (gdrcopy_ADDED)
    message(STATUS "gdrcopy was not installed. Building from Source")

    if (MORPHEUS_TP_INSTALL_DOCA_DEPS)
      set(gdrcopy_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
    else()
      set(gdrcopy_INSTALL_DIR ${gdrcopy_BINARY_DIR}/install)
    endif()

    # Get the Compiler settings to forward onto autoconf
    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_UC)
    set(COMPILER_SETTINGS
        "CXX=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_CXX_COMPILER}"
        "CPP=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER} -E"
        "CC=${CMAKE_C_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER}"
        "AR=${CMAKE_C_COMPILER_AR}"
        "RANLIB=${CMAKE_C_COMPILER_RANLIB}"
        "NM=${CMAKE_NM}"
        "STRIP=${CMAKE_STRIP}"
        "CFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}}"
        "CPPFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}}"
        "CXXFLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UC}}"
        "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE_UC}}"
    )

    ExternalProject_Add(gdrcopy
      PREFIX              ${gdrcopy_BINARY_DIR} # Root directory for bsd
      SOURCE_DIR          ${gdrcopy_BINARY_DIR} # Move source over from cpm dir and build in binary dir
      INSTALL_DIR         ${gdrcopy_INSTALL_DIR}

      DOWNLOAD_COMMAND    ${CMAKE_COMMAND} -E copy_directory ${gdrcopy_SOURCE_DIR} ${gdrcopy_BINARY_DIR}

      CONFIGURE_COMMAND   ""

      BUILD_COMMAND       make -j lib lib_install
      BUILD_IN_SOURCE     TRUE
      BUILD_BYPRODUCTS    <INSTALL_DIR>/lib/libgdrapi.so

      INSTALL_COMMAND     make prefix=<INSTALL_DIR> lib_install

      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
    )
  endif()
endfunction()

find_and_configure_gdrcopy(${MORPHEUS_TP_GDRCOPY_VERSION})

LIST(POP_BACK CMAKE_MESSAGE_CONTEXT)