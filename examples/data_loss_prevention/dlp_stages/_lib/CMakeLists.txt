# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "dlp_regex_processor")

if(PROJECT_IS_TOP_LEVEL)
  find_library(MORPHEUS morpheus REQUIRED)
else()
  set(MORPHEUS morpheus)
endif()

morpheus_add_pybind11_module(regex_processor
  SOURCE_FILES
    "regex_processor.cpp"
  INCLUDE_DIRS
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  LINK_TARGETS
    ${MORPHEUS}
    CUDA::nvtx3
    cudf::cudf
    glog::glog
)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
