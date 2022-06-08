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

function(find_and_configure_srf version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "srf")

  rapids_cpm_find(srf ${version}
    GLOBAL_TARGETS
      srf::srf srf::pysrf
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-exports
    CPM_ARGS
      GIT_REPOSITORY  git@github.com:nv-morpheus/SRF.git
      GIT_TAG         main
      GIT_SHALLOW     TRUE
      OPTIONS         "SRF_BUILD_EXAMPLES OFF"
                      "SRF_BUILD_TESTS OFF"
                      "SRF_BUILD_BENCHMARKS OFF"
                      "SRF_BUILD_PYTHON ON"
                      "SRF_ENABLE_XTENSOR ON"
                      "SRF_ENABLE_MATX ON"
                      "SRF_USE_CONDA ${MORPHEUS_USE_CONDA}"
                      "SRF_USE_CCACHE ${MORPHEUS_USE_CCACHE}"
                      "SRF_USE_CLANG_TIDY ${MORPHEUS_USE_CLANG_TIDY}"
                      "SRF_PYTHON_INPLACE_BUILD ${MORPHEUS_PYTHON_INPLACE_BUILD}"
                      "RMM_VERSION ${RAPIDS_VERSION}"
  )

  if(srf_ADDED)

    # Now ensure its installed
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # detect virtualenv and set Pip args accordingly
    if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
      set(_pip_args)
    else()
      set(_pip_args "--user")
    endif()

    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
      list(APPEND _pip_args "-e")
    endif()

    add_custom_command(
      OUTPUT ${srf_BINARY_DIR}/python/srf.egg-info/PKG-INFO
      COMMAND ${Python3_EXECUTABLE} -m pip install ${_pip_args} ${srf_BINARY_DIR}/python
      DEPENDS srf_python_rebuild
      COMMENT "Installing srf python package"
    )

    add_custom_target(
      install_srf_python ALL
      DEPENDS ${srf_BINARY_DIR}/python/srf.egg-info/PKG-INFO
    )
  endif()
endfunction()

find_and_configure_srf(${SRF_VERSION})
