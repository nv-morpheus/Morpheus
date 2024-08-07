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
#include "morpheus/messages/multi.hpp"                 // for MultiMessage
#include "morpheus/messages/multi_inference.hpp"       // for MultiInferenceMessage
#include "morpheus/objects/tensor_object.hpp"          // for TensorObject
#include "morpheus/stages/preprocess_fil.hpp"          // for PreprocessFILStage, PreprocessFILStageCC, PreprocessFI...
#include "morpheus/types.hpp"                          // for TensorIndex

#include <cuda_runtime.h>       // for cudaMemcpy, cudaMemcpyKind
#include <gtest/gtest.h>        // for EXPECT_EQ, Message, TestPartResult, TestInfo, TEST_F
#include <mrc/cuda/common.hpp>  // for __check_cuda_errors, MRC_CHECK_CUDA
#include <pybind11/gil.h>       // for gil_scoped_release

#include <filesystem>  // for path, operator/
#include <memory>      // for allocator, make_shared, __shared_ptr_access, shared_ptr
#include <string>      // for string
#include <utility>     // for move
#include <vector>      // for vector

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(PreprocessFIL);

TEST_F(TestPreprocessFIL, TestProcessControlMessageAndMultiMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "float_str.csv";

    // Create a dataframe from a file
    auto cm_table = load_table_from_file(input_file);
    auto cm_meta  = MessageMeta::create_from_cpp(std::move(cm_table));

    auto mm_table = load_table_from_file(input_file);
    auto mm_meta  = MessageMeta::create_from_cpp(std::move(mm_table));

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(cm_meta);

    // Create PreProcessControlMessageStage
    auto cm_stage    = std::make_shared<PreprocessFILStageCM>(std::vector<std::string>{"float_str1", "float_str2"});
    auto cm_response = cm_stage->on_data(cm);

    // Create MultiMessage
    auto mm = std::make_shared<MultiMessage>(mm_meta);
    // Create PreProcessMultiMessageStage
    auto mm_stage    = std::make_shared<PreprocessFILStageMM>(std::vector<std::string>{"float_str1", "float_str2"});
    auto mm_response = mm_stage->on_data(mm);

    auto cm_tensors = cm_response->tensors();
    auto mm_tensors = mm_response->memory;

    // Verify output tensors
    std::vector<float> expected_input__0 = {1, 4, 2, 5, 3, 6};
    auto cm_input__0                     = cm_tensors->get_tensor("input__0");
    auto mm_input__0                     = mm_tensors->get_tensor("input__0");
    std::vector<float> cm_input__0_host(cm_input__0.count());
    std::vector<float> mm_input__0_host(mm_input__0.count());
    MRC_CHECK_CUDA(cudaMemcpy(
        cm_input__0_host.data(), cm_input__0.data(), cm_input__0.count() * sizeof(float), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(
        mm_input__0_host.data(), mm_input__0.data(), mm_input__0.count() * sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected_input__0, cm_input__0_host);
    EXPECT_EQ(cm_input__0_host, mm_input__0_host);

    std::vector<TensorIndex> expected_seq_ids = {0, 0, 1, 1, 0, 1, 2, 0, 1};
    auto cm_seq_ids                           = cm_tensors->get_tensor("seq_ids");
    auto mm_seq_ids                           = mm_tensors->get_tensor("seq_ids");
    std::vector<TensorIndex> cm_seq_ids_host(cm_seq_ids.count());
    std::vector<TensorIndex> mm_seq_ids_host(mm_seq_ids.count());
    MRC_CHECK_CUDA(cudaMemcpy(
        cm_seq_ids_host.data(), cm_seq_ids.data(), cm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(cudaMemcpy(
        mm_seq_ids_host.data(), mm_seq_ids.data(), mm_seq_ids.count() * sizeof(TensorIndex), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected_seq_ids, cm_seq_ids_host);
    EXPECT_EQ(cm_seq_ids_host, mm_seq_ids_host);
}
