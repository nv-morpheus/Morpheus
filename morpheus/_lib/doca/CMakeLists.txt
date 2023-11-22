# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

# Need to load this first before calling find_package(doca REQUIRED)
morpheus_configure_libmd()
morpheus_configure_libbsd()


set(doca_ROOT "/opt/mellanox/doca")
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(doca REQUIRED)
set(CMAKE_FIND_DEBUG_MODE FALSE)

add_library(morpheus_doca
  # Keep these sorted!
  src/doca_context.cpp
  src/doca_rx_pipe.cpp
  src/doca_rx_queue.cpp
  src/doca_semaphore.cpp
  src/doca_source_kernels.cu
  src/doca_source.cpp
  src/rte_context.cpp
)

add_library(${PROJECT_NAME}::morpheus_doca ALIAS morpheus_doca)

target_include_directories(morpheus_doca
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(morpheus_doca
  PRIVATE
    doca::doca
  PUBLIC
    ${PROJECT_NAME}::morpheus

)

# Ideally, we dont use glob here. But there is no good way to guarantee you dont miss anything like *.cpp
file(GLOB_RECURSE morpheus_doca_public_headers
  LIST_DIRECTORIES FALSE
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# Add headers to target sources file_set so they can be installed
# https://discourse.cmake.org/t/installing-headers-the-modern-way-regurgitated-and-revisited/3238/3
target_sources(morpheus_doca
  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    FILES
      ${morpheus_doca_public_headers}
)

target_compile_definitions(morpheus_doca
  PUBLIC
    DOCA_ALLOW_EXPERIMENTAL_API
)

set_target_properties(morpheus_doca
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

install(
  TARGETS
    morpheus_doca
  EXPORT
    ${PROJECT_NAME}-exports
  FILE_SET
    public_headers
  COMPONENT
    Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus_doca ${CMAKE_CURRENT_SOURCE_DIR})
endif()
