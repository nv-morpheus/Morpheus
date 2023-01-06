# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ######################################################################################################################
# * CMake properties ------------------------------------------------------------------------------

list(APPEND CMAKE_MESSAGE_CONTEXT "compiler")

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# ###################################
# - Compiler Flags -----------------
check_c_compiler_flag("-O0" COMPILER_C_HAS_O0)

if(COMPILER_C_HAS_O0)
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
endif()

check_cxx_compiler_flag("-O0" COMPILER_CXX_HAS_O0)

if(COMPILER_CXX_HAS_O0)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

  # Also set cuda here
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0")
endif()

# ###################################
# - Compiler Checks ----------------

# Only set the clang-tidy options for our source code targets
if(MORPHEUS_USE_CLANG_TIDY)
  set(CMAKE_C_CLANG_TIDY "clang-tidy")
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
  message(STATUS "Enabling clang-tidy for targets in project ${PROJECT_NAME}")
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
