#=============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

function(find_and_configure_mrc version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "mrc")

  rapids_cpm_find(mrc ${version} CONFIG REQUIRED
    GLOBAL_TARGETS
      mrc::mrc mrc::pymrc
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      #GIT_REPOSITORY  https://github.com/nv-morpheus/MRC.git
      GIT_REPOSITORY  /home/drobison/Development/devin-mrc-public
      GIT_SUBMODULES  utilities
      #GIT_TAG         branch-${version}
      GIT_TAG         devin-issue-225
      GIT_SHALLOW     TRUE
      OPTIONS         "MRC_BUILD_EXAMPLES OFF"
                      "MRC_BUILD_TESTS OFF"
                      "MRC_BUILD_BENCHMARKS OFF"
                      "MRC_BUILD_PYTHON ON"
                      "MRC_ENABLE_XTENSOR ON"
                      "MRC_ENABLE_MATX ON"
                      "MRC_USE_CONDA ${MORPHEUS_USE_CONDA}"
                      "MRC_USE_CCACHE ${MORPHEUS_USE_CCACHE}"
                      "MRC_USE_CLANG_TIDY ${MORPHEUS_USE_CLANG_TIDY}"
                      "MRC_PYTHON_INPLACE_BUILD OFF"
                      "MRC_PYTHON_PERFORM_INSTALL ON"
                      "MRC_PYTHON_BUILD_STUBS ${MORPHEUS_BUILD_PYTHON_STUBS}"
                      "RMM_VERSION ${MORPHEUS_RAPIDS_VERSION}"
  )

endfunction()

find_and_configure_MRC(${MRC_VERSION})
