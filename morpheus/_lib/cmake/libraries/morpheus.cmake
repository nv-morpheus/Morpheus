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

message(STATUS "Adding library: morpheus")

include(GenerateExportHeader)

# Place the two cuda sources in their own target and disable IWYU for that target.
add_library(cuda_objs
  OBJECT
    ${MORPHEUS_LIB_ROOT}/src/utilities/matx_util.cu
)

set_target_properties(
  cuda_objs
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    C_INCLUDE_WHAT_YOU_USE ""
    CXX_INCLUDE_WHAT_YOU_USE ""
    EXPORT_COMPILE_COMMANDS OFF
)

target_include_directories(cuda_objs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(cuda_objs
  PUBLIC
    cudf::cudf
    matx::matx
)


add_library(morpheus
    # Keep these sorted!
    ${MORPHEUS_LIB_ROOT}/src/io/deserializers.cpp
    ${MORPHEUS_LIB_ROOT}/src/io/serializers.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/response_memory_probs.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/response_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/tensor_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/meta.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_response_probs.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_response.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/data_table.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/dev_mem_info.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/dtype.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/fiber_queue.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/file_types.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/memory_descriptor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/mutable_table_ctx_mgr.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/python_data_table.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/rmm_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/table_info.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/tensor_object.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/wrapped_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_classification.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_scores_stage_base.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_scores.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/deserialize.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/file_source.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/filter_detection.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/kafka_source.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/preallocate.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/preprocess_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/preprocess_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/serialize.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/triton_inference.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/write_to_file.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/cudf_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/cupy_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/python_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/string_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/table_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/tensor_util.cpp
    $<TARGET_OBJECTS:cuda_objs>
)

add_library(${PROJECT_NAME}::morpheus ALIAS morpheus)

target_link_libraries(morpheus
  PUBLIC
    ${cudf_helpers_target}
    CUDA::nvToolsExt
    morpheus_utils
    mrc::pymrc
    RDKAFKA::RDKAFKA
    TritonClient::httpclient_static
)

target_include_directories(morpheus
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# We want to use RUNPATH instead of RPATH to allow LD_LIBRARY_PATH to take precedence over the paths specified in the
# binary. This is necessary to allow ld to find the real libcuda.so instead of the stub. Eventually, this can be removed
# once upgraded to cuda-python 12.1. Ideally, cuda-python would just load libcuda.so.1 which would take precedence over
# libcuda.so. Relavant issue: https://github.com/NVIDIA/cuda-python/issues/17
target_link_options(morpheus PUBLIC "-Wl,--enable-new-dtags")

set_target_properties(morpheus
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Generates an include file for specifying external linkage since everything is hidden by default
generate_export_header(morpheus
  NO_EXPORT_MACRO_NAME MORPHEUS_LOCAL
)

install(
  TARGETS
    morpheus
  EXPORT
    ${PROJECT_NAME}-exports
  COMPONENT
    Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus ${CMAKE_CURRENT_SOURCE_DIR})
endif()
