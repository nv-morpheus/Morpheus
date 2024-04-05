/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../test_utils/common.hpp"  // for get_morpheus_root, TEST_CLASS, morpheus

#include "morpheus/io/deserializers.hpp"               // for load_table_from_file
#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/messages/multi_response.hpp"        // for MultiResponseMessage
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/table_info.hpp"             // for TableInfo
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/stages/add_scores.hpp"              // for AddScoresStage
#include "morpheus/types.hpp"                          // for TensorIndex

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column_view.hpp>  // for column_view
#include <gtest/gtest.h>                // for EXPECT_EQ, Message, TestInfo, TestPartResult, TEST_F
#include <mrc/cuda/common.hpp>          // for __check_cuda_errors, MRC_CHECK_CUDA
#include <pybind11/gil.h>               // for gil_scoped_release
#include <rmm/cuda_stream_view.hpp>     // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>        // for device_buffer

#include <cstddef>     // for size_t
#include <filesystem>  // for operator/, path
#include <map>         // for map
#include <memory>      // for make_shared, allocator, __shared_ptr_access, shared_ptr
#include <string>      // for string
#include <utility>     // for move
#include <vector>      // for vector

using namespace morpheus;

TEST_CLASS(AddScores);

TEST_F(TestAddScores, TestProcessControlMessageAndMultiResponseMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "floats.csv";

    TensorIndex cols_size  = 1;
    TensorIndex mess_count = 3;
    auto packed_data =
        std::make_shared<rmm::device_buffer>(cols_size * mess_count * sizeof(double), rmm::cuda_stream_per_thread);

    // Create a dataframe from a file
    auto table = load_table_from_file(input_file);
    auto meta  = MessageMeta::create_from_cpp(std::move(table));

    std::map<std::size_t, std::string> idx2label = {{0, "float"}};

    // Create MultiResponseMessage
    auto tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto tensor_memory = std::make_shared<TensorMemory>(mess_count);
    tensor_memory->set_tensor("probs", std::move(tensor));
    auto mm = std::make_shared<MultiResponseMessage>(meta, 0, mess_count, std::move(tensor_memory));

    // Create PreProcessMultiMessageStage
    auto mm_stage    = std::make_shared<AddScoresStage<MultiResponseMessage, MultiResponseMessage>>(idx2label);
    auto mm_response = mm_stage->on_data(mm);

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);
    auto cm_tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto cm_tensor_memory = std::make_shared<TensorMemory>(mess_count);
    cm_tensor_memory->set_tensor("probs", std::move(cm_tensor));
    cm->tensors(cm_tensor_memory);

    // Create PreProcessControlMessageStage
    auto cm_stage    = std::make_shared<AddScoresStage<ControlMessage, ControlMessage>>(idx2label);
    auto cm_response = cm_stage->on_data(cm);

    // Verify the output meta
    std::vector<float> expected_meta = {0, 0, 1.4013e-45};
    auto mm_meta                     = mm_response->get_meta().get_column(0);
    auto cm_meta                     = cm_response->payload()->get_info().get_column(0);
    std::vector<float> mm_meta_host(mm_meta.size());
    std::vector<float> cm_meta_host(cm_meta.size());
    MRC_CHECK_CUDA(cudaMemcpy(
        mm_meta_host.data(), mm_meta.data<double>(), mm_meta.size() * sizeof(double), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(
        cm_meta_host.data(), cm_meta.data<double>(), cm_meta.size() * sizeof(double), cudaMemcpyDeviceToHost));
    EXPECT_EQ(mm_meta_host, expected_meta);
    EXPECT_EQ(mm_meta_host, cm_meta_host);
}
