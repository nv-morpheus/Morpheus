#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

function(find_and_configure_rapidjson)

  list(APPEND CMAKE_MESSAGE_CONTEXT "rapidjson")

  rapids_cpm_find(rapidjson 1.1.0
      GLOBAL_TARGETS
        rapidjson rapidjson::rapidjson
      BUILD_EXPORT_SET
        ${PROJECT_NAME}-exports
      INSTALL_EXPORT_SET
        ${PROJECT_NAME}-exports
      CPM_ARGS
        GIT_REPOSITORY    https://github.com/Tencent/rapidjson.git
        GIT_TAG           v1.1.0
        OPTIONS           RAPIDJSON_BUILD_DOC OFF
                          RAPIDJSON_BUILD_EXAMPLES OFF
                          RAPIDJSON_BUILD_TESTS OFF
  )

endfunction()

find_and_configure_rapidjson()
