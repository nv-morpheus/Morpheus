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

function(find_and_configure_libcudacxx VERSION)
  ###  NOTE: pulled version is set in rapids_cpm_package_overrides.json ###
  list(APPEND CMAKE_MESSAGE_CONTEXT "libcudacxx")

  # Use rapids-cpm to load libcudacxx. This makes an interface library
  # libcudacxx::libcudacxx that you can link against. If rapids_cpm_libcudaxx is
  # removed, be sure to set `libcudacxx_SOURCE_DIR` since other libraries can
  # depend on this variable. Set it in the parent scope to ensure its valid
  # See: https://github.com/rapidsai/rapids-cmake/issues/117

  # Requires RAPIDS ver >= 21.12
  include("${rapids-cmake-dir}/cpm/libcudacxx.cmake")
  rapids_cpm_libcudacxx(
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
  )

endfunction()

find_and_configure_libcudacxx(${LIBCUDACXX_VERSION})
