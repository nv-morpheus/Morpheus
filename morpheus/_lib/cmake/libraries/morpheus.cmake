# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

add_library(morpheus
    # Keep these sorted!
    ${MORPHEUS_LIB_ROOT}/src/io/serializers.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/inference_memory_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/response_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/response_memory_probs.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/memory/tensor_memory.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/meta.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_inference_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_response.cpp
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_response_probs.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/fiber_queue.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/file_types.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/wrapped_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/python_data_table.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/rmm_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/table_info.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_classification.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_scores.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/deserialize.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/file_source.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/filter_detection.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/kafka_source.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/preprocess_fil.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/preprocess_nlp.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/serialize.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/triton_inference.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/write_to_file.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/cudf_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/cupy_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/string_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/table_util.cpp
)

add_library(${PROJECT_NAME}::morpheus ALIAS morpheus)

target_link_libraries(morpheus
    PUBLIC
      ${cudf_helpers_target}
      TritonClient::httpclient_static
      RDKAFKA::RDKAFKA
)

target_include_directories(morpheus
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

set_target_properties(morpheus PROPERTIES CXX_VISIBILITY_PRESET hidden)

message(STATUS " Install dest: (morpheus) ${MORPHEUS_LIB_INSTALL_DIR}")
install(
    TARGETS
      morpheus
    EXPORT
      ${PROJECT_NAME}-exports
    LIBRARY DESTINATION
      "${MORPHEUS_LIB_INSTALL_DIR}"
    COMPONENT Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  inplace_build_copy(morpheus ${CMAKE_CURRENT_SOURCE_DIR})
endif()
