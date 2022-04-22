#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

function(find_and_configure_matx version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "matx")

  if(CUDAToolkit_FOUND AND (CUDAToolkit_VERSION VERSION_GREATER "11.4"))

    rapids_cpm_find(matx ${version}
      GLOBAL_TARGETS
        matx matx::matx
      BUILD_EXPORT_SET
        ${PROJECT_NAME}-exports
      INSTALL_EXPORT_SET
        ${PROJECT_NAME}-exports
      CPM_ARGS
        GIT_REPOSITORY  https://github.com/NVIDIA/MatX.git
        GIT_TAG         "v${version}"
        GIT_SHALLOW     TRUE
        PATCH_COMMAND   git checkout -- . && git apply --whitespace=fix ${PROJECT_SOURCE_DIR}/cmake/deps/patches/matx.patch
        OPTIONS         "BUILD_EXAMPLES OFF"
                        "BUILD_TESTS OFF"
                        "MATX_INSTALL ON"
    )

  else()
    message(SEND_ERROR "Unable to add MatX dependency. CUDA Version must be greater than 11.4. Current CUDA Version: ${CUDAToolkit_VERSION}")
  endif()

endfunction()

find_and_configure_matx(${MATX_VERSION})
