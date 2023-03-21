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

add_cython_target(cudf_helpers "${MORPHEUS_LIB_ROOT}/cudf_helpers.pyx"
  CXX
)

add_custom_target(cudf_helpers_gen DEPENDS ${cudf_helpers})

# This target generates headers used by other parts of the code base.
# The C++ checks used in CI need these headers but don't require an actual build.
# The `morpheus_style_checks` target allows these to be generated without a full build of Morpheus.
add_dependencies(${PROJECT_NAME}_style_checks cudf_helpers_gen)

# Place the two cuda sources in their own target and disable IWYU for that target.
add_library(morpheus_objs
  OBJECT
    ${cudf_helpers}
    ${MORPHEUS_LIB_ROOT}/src/utilities/matx_util.cu
)

set_target_properties(
  morpheus_objs
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    C_INCLUDE_WHAT_YOU_USE ""
    CXX_INCLUDE_WHAT_YOU_USE ""
    EXPORT_COMPILE_COMMANDS OFF
)

target_include_directories(morpheus_objs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(morpheus_objs
  PUBLIC
    cudf::cudf
    matx::matx
    Python3::Module
    Python3::NumPy
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
    $<TARGET_OBJECTS:morpheus_objs>
)

add_library(${PROJECT_NAME}::morpheus ALIAS morpheus)

target_link_libraries(morpheus
  PUBLIC
    CUDA::nvToolsExt
    cudf::cudf
    mrc::pymrc
    pybind11::pybind11
    RDKAFKA::RDKAFKA
    TritonClient::httpclient_static
)

target_include_directories(morpheus
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

set_target_properties(morpheus
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

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
