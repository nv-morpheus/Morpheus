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

set(DOCA_BUILD_FILES "")
if(MORPHEUS_SUPPORT_DOCA)
  set(DOCA_BUILD_FILES
    # ${MORPHEUS_LIB_ROOT}/src/doca/samples/common.c
    # ${MORPHEUS_LIB_ROOT}/src/doca/doca_context.cpp
    # ${MORPHEUS_LIB_ROOT}/src/doca/dpdk_utils.c
    # ${MORPHEUS_LIB_ROOT}/src/doca/flows.c
    # ${MORPHEUS_LIB_ROOT}/src/doca/gpu_init.c
    # ${MORPHEUS_LIB_ROOT}/src/doca/offload_rules.c
    # ${MORPHEUS_LIB_ROOT}/src/doca/utils.c
    ${MORPHEUS_LIB_ROOT}/src/stages/doca_source.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/doca_source_kernels.cu
  )
endif()

add_library(morpheus
    # Keep these sorted!
    ${MORPHEUS_LIB_ROOT}/src/io/deserializers.cpp
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
    ${MORPHEUS_LIB_ROOT}/src/messages/multi_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/fiber_queue.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/file_types.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/mutable_table_ctx_mgr.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/wrapped_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/python_data_table.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/rmm_tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/objects/tensor.cpp
    ${MORPHEUS_LIB_ROOT}/src/stages/add_classification.cpp
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
    ${MORPHEUS_LIB_ROOT}/src/utilities/string_util.cpp
    ${MORPHEUS_LIB_ROOT}/src/utilities/table_util.cpp
    ${DOCA_BUILD_FILES}
)

add_library(${PROJECT_NAME}::morpheus ALIAS morpheus)

target_link_libraries(morpheus
    PUBLIC
      cuda_utils
      ${cudf_helpers_target}
      TritonClient::httpclient_static
      RDKAFKA::RDKAFKA
)

target_include_directories(morpheus
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
)

if(MORPHEUS_SUPPORT_DOCA)
    target_link_libraries(morpheus
        PUBLIC
            -L/opt/mellanox/dpdk/lib/x86_64-linux-gnu/
            -L/opt/mellanox/doca/lib/x86_64-linux-gnu/
        #   -L/home/charris/dev/doca-testing/doca/install/lib/x86_64-linux-gnu
        #   -L/home/charris/dev/doca-testing/dpdk-doca-gpu/install/lib/x86_64-linux-gnu
            libdoca_argp.so
            libdoca_common.so
            libdoca_gpu.so
            libdoca_gpu_device.so
            libdoca_flow.so
            librte_bus_auxiliary.so
            librte_bus_pci.so
            librte_bus_vdev.so
            librte_common_mlx5.so
            librte_eal.so
            librte_ethdev.so
            librte_gpudev.so
            librte_hash.so
            librte_ip_frag.so
            librte_kvargs.so
            librte_mbuf.so
            librte_mempool.so
            librte_meter.so
            librte_net_mlx5.so
            librte_net.so
            librte_pci.so
            librte_rcu.so
            librte_ring.so
            librte_telemetry.so
    )
    target_include_directories(morpheus
        PUBLIC
            /opt/mellanox/doca/include/
            /opt/mellanox/dpdk/include/dpdk
            /opt/mellanox/dpdk/include/x86_64-linux-gnu/dpdk
            /usr/local/include
    )
endif()

set_target_properties(morpheus PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(morpheus PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(morpheus PROPERTIES CUDA_STANDARD 17)

install(
    TARGETS
      morpheus
    EXPORT
      ${PROJECT_NAME}-exports
    COMPONENT Wheel
)

if (MORPHEUS_PYTHON_INPLACE_BUILD)
  morpheus_utils_inplace_build_copy(morpheus ${CMAKE_CURRENT_SOURCE_DIR})
endif()
