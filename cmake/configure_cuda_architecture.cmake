# SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Conda / Vcpkg configuration
list(APPEND CMAKE_MESSAGE_CONTEXT "toolchain")
message(STATUS "Configuring CUDA Architecture")
# Default to using "" for CUDA_ARCHITECTURES to build based on GPU in system
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES was not defined. Defaulting to '' to build only for local architecture. "
                 "Specify -DCMAKE_CUDA_ARCHITECTURES='ALL' to build for all archs.")
endif()

# Initialize CUDA architectures
rapids_cuda_init_architectures(morpheus)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)
