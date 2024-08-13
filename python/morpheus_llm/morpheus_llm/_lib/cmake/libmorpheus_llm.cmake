# =============================================================================
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

add_library(morpheus_llm

  # Keep these sorted!
  src/llm/input_map.cpp
  src/llm/llm_context.cpp
  src/llm/llm_engine.cpp
  src/llm/llm_node_runner.cpp
  src/llm/llm_node.cpp
  src/llm/llm_task_handler_runner.cpp
  src/llm/llm_task.cpp
  src/llm/utils.cpp
)

add_library(${PROJECT_NAME}::morpheus_llm ALIAS morpheus_llm)

# fixme check if all these are needed
target_link_libraries(morpheus_llm
  PRIVATE
    matx::matx
    $<$<CONFIG:Debug>:ZLIB::ZLIB>
  PUBLIC
    $<TARGET_NAME_IF_EXISTS:conda_env>
    cudf::cudf
    CUDA::nvtx3
    mrc::pymrc
    ${PROJECT_NAME}::morpheus

)

# fixme: find another way to include morpheus headers
target_include_directories(morpheus_llm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/morpheus/_lib/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# In debug mode, dont allow missing symbols
target_link_options(morpheus_llm PUBLIC "$<$<CONFIG:Debug>:-Wl,--no-allow-shlib-undefined>")

# Ideally, we dont use glob here. But there is no good way to guarantee you dont miss anything like *.cpp
file(GLOB_RECURSE morpheus_llm_public_headers
  LIST_DIRECTORIES FALSE
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/morpheus/*"
)

# Add headers to target sources file_set so they can be installed
# https://discourse.cmake.org/t/installing-headers-the-modern-way-regurgitated-and-revisited/3238/3
target_sources(morpheus_llm
  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES
  ${morpheus_llm_public_headers}
)


# We want to use RUNPATH instead of RPATH to allow LD_LIBRARY_PATH to take precedence over the paths specified in the
# binary. This is necessary to allow ld to find the real libcuda.so instead of the stub. Eventually, this can be removed
# once upgraded to cuda-python 12.1. Ideally, cuda-python would just load libcuda.so.1 which would take precedence over
# libcuda.so. Relavant issue: https://github.com/NVIDIA/cuda-python/issues/17
target_link_options(morpheus_llm PUBLIC "-Wl,--enable-new-dtags")

# required to link code containing pybind11 headers
target_link_options(morpheus_llm PUBLIC "-Wl,--gc-sections")

set_target_properties(morpheus_llm
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus_llm ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ##################################################################################################
# - install targets --------------------------------------------------------------------------------

# Get the library directory in a cross-platform way
rapids_cmake_install_lib_dir(lib_dir)

include(CPack)
include(GNUInstallDirs)

install(
    TARGETS
      morpheus_llm
    EXPORT
      ${PROJECT_NAME}-core-exports
    LIBRARY
    DESTINATION ${lib_dir}
    FILE_SET
      public_headers
)
