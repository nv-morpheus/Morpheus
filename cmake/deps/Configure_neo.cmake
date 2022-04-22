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

function(find_and_configure_neo version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "neo")

  # Check if the developer has specified a specific remote or local version of Neo
  if (DEFINED CACHE{NEO_GIT_REPOSITORY} OR DEFINED CACHE{CPM_neo_SOURCE})
    rapids_cpm_find(neo ${version}
      GLOBAL_TARGETS
        neo::neo neo::pyneo
      BUILD_EXPORT_SET
        ${PROJECT_NAME}-exports
      INSTALL_EXPORT_SET
        ${PROJECT_NAME}-exports
      CPM_ARGS
        GIT_REPOSITORY  ${NEO_GIT_REPOSITORY}
        GIT_TAG         branch-${version}
        GIT_SHALLOW     TRUE
        OPTIONS         "NEO_BUILD_EXAMPLES OFF"
                        "NEO_BUILD_TESTS OFF"
                        "NEO_BUILD_BENCHMARKS OFF"
                        "NEO_BUILD_PYTHON ON"
                        "NEO_ENABLE_XTENSOR ON"
                        "NEO_ENABLE_MATX ON"
                        "NEO_USE_CONDA ${MORPHEUS_USE_CONDA}"
                        "NEO_USE_CCACHE ${MORPHEUS_USE_CCACHE}"
                        "NEO_USE_CLANG_TIDY ${MORPHEUS_USE_CLANG_TIDY}"
                        "NEO_PYTHON_INPLACE_BUILD ${MORPHEUS_PYTHON_INPLACE_BUILD}"
                        "RMM_VERSION ${RAPIDS_VERSION}"
    )
  else()
    rapids_find_package(neo REQUIRED
      GLOBAL_TARGETS
        neo::neo neo::pyneo
      BUILD_EXPORT_SET
        ${PROJECT_NAME}-exports
      INSTALL_EXPORT_SET
        ${PROJECT_NAME}-exports
      FIND_ARGS
        ${version}
    )
  endif()

  if(neo_ADDED)

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
      OUTPUT ${neo_BINARY_DIR}/python/neo.egg-info/PKG-INFO
      COMMAND ${Python3_EXECUTABLE} -m pip install ${_pip_args} ${neo_BINARY_DIR}/python
      DEPENDS neo_python_rebuild
      COMMENT "Installing neo python package"
    )

    add_custom_target(
      install_neo_python ALL
      DEPENDS ${neo_BINARY_DIR}/python/neo.egg-info/PKG-INFO
    )
  endif()
endfunction()

find_and_configure_neo(${NEO_VERSION})
