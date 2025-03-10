# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

# Disable the assert that checks for the GIL being held during incref/decref
# https://pybind11.readthedocs.io/en/stable/changelog.html#version-2-11-1-july-17-2023
add_compile_definitions("PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF=1")

if (MORPHEUS_BUILD_MORPHEUS_CORE)
  add_subdirectory(morpheus)
endif()

if (MORPHEUS_BUILD_MORPHEUS_LLM)
  add_subdirectory(morpheus_llm)
endif()

if (MORPHEUS_BUILD_MORPHEUS_DFP)
  add_subdirectory(morpheus_dfp)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
