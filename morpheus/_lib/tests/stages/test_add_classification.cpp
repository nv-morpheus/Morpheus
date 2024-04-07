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

#include "morpheus/messages/control.hpp"               // for ControlMessage
#include "morpheus/messages/memory/tensor_memory.hpp"  // for TensorMemory
#include "morpheus/messages/meta.hpp"                  // for MessageMeta
#include "morpheus/messages/multi_response.hpp"        // for MultiResponseMessage
#include "morpheus/objects/dtype.hpp"                  // for DType
#include "morpheus/objects/table_info.hpp"             // for TableInfo
#include "morpheus/objects/tensor.hpp"                 // for Tensor
#include "morpheus/stages/add_classification.hpp"      // for AddClassificationsStage
#include "morpheus/types.hpp"                          // for TensorIndex

#include <cuda_runtime.h>                      // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column_view.hpp>         // for column_view
#include <cudf/io/csv.hpp>                     // for csv_reader_options_builder, read_csv, csv_reader_options
#include <cudf/io/types.hpp>                   // for source_info, table_with_metadata
#include <cudf/types.hpp>                      // for data_type
#include <cudf/utilities/type_dispatcher.hpp>  // for type_to_id
#include <gtest/gtest.h>                       // for EXPECT_EQ, Message, TestInfo, TestPartResult, TEST_F
#include <mrc/cuda/common.hpp>                 // for __check_cuda_errors, MRC_CHECK_CUDA
#include <pybind11/gil.h>                      // for gil_scoped_release
#include <rmm/cuda_stream_view.hpp>            // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>               // for device_buffer

#include <cstddef>     // for size_t
#include <cstdint>     // for uint8_t
#include <filesystem>  // for operator/, path
#include <map>         // for map
#include <memory>      // for make_shared, allocator, __shared_ptr_access, shared_ptr
#include <string>      // for string
#include <utility>     // for move
#include <vector>      // for vector

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(AddClassification);

template <typename T>
auto convert_to_host(rmm::device_buffer& buffer)
{
    std::vector<T> host_buffer(buffer.size() / sizeof(T));

    MRC_CHECK_CUDA(cudaMemcpy(host_buffer.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost));

    return host_buffer;
}

TEST_F(TestAddClassification, TestProcessControlMessageAndMultiResponseMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "bools.csv";

    TensorIndex cols_size  = 3;
    TensorIndex mess_count = 3;

    double threshold = 0.4;

    auto packed_data_host = std::vector<double>{
        0.1,
        0.2,
        0.3,  // All below
        0.5,
        0.0,
        0.0,  // Only one above
        0.7,
        0.1,
        0.9  // All above
    };

    auto packed_data = std::make_shared<rmm::device_buffer>(
        packed_data_host.data(), cols_size * mess_count * sizeof(double), rmm::cuda_stream_per_thread);

    cudf::io::csv_reader_options read_opts = cudf::io::csv_reader_options::builder(cudf::io::source_info(input_file))
                                                 .dtypes({cudf::data_type(cudf::data_type{cudf::type_to_id<bool>()})})
                                                 .header(0);
    auto meta_mm = MessageMeta::create_from_cpp(cudf::io::read_csv(read_opts));

    std::map<std::size_t, std::string> idx2label = {{0, "bool"}};

    // Create MultiResponseMessage
    auto tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto tensor_memory = std::make_shared<TensorMemory>(mess_count);
    tensor_memory->set_tensor("probs", std::move(tensor));
    auto mm = std::make_shared<MultiResponseMessage>(std::move(meta_mm), 0, mess_count, std::move(tensor_memory));

    // Create PreProcessMultiMessageStage
    auto mm_stage    = std::make_shared<AddClassificationsStageMM>(idx2label, 0.4);
    auto mm_response = mm_stage->on_data(mm);

    // Create a separate dataframe from a file (otherwise they will overwrite eachother)
    auto meta_cm = MessageMeta::create_from_cpp(cudf::io::read_csv(read_opts));

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(std::move(meta_cm));
    auto cm_tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto cm_tensor_memory = std::make_shared<TensorMemory>(mess_count);
    cm_tensor_memory->set_tensor("probs", std::move(cm_tensor));
    cm->tensors(cm_tensor_memory);

    // Create PreProcessControlMessageStage
    auto cm_stage    = std::make_shared<AddClassificationsStageCM>(idx2label, 0.4);
    auto cm_response = cm_stage->on_data(cm);

    // Verify the output meta
    std::vector<uint8_t> expected_meta = {'\0', '\x1', '\x1'};
    auto mm_meta                       = mm_response->get_meta().get_column(0);
    auto cm_meta                       = cm_response->payload()->get_info().get_column(0);

    // std::vector<bool> is a template specialization which does not have data() method, use std::vector<uint8_t> here
    std::vector<uint8_t> mm_meta_host(mm_meta.size());
    std::vector<uint8_t> cm_meta_host(cm_meta.size());
    MRC_CHECK_CUDA(
        cudaMemcpy(mm_meta_host.data(), mm_meta.data<bool>(), mm_meta.size() * sizeof(bool), cudaMemcpyDeviceToHost));
    MRC_CHECK_CUDA(
        cudaMemcpy(cm_meta_host.data(), cm_meta.data<bool>(), cm_meta.size() * sizeof(bool), cudaMemcpyDeviceToHost));
    EXPECT_EQ(mm_meta_host, expected_meta);
    EXPECT_EQ(mm_meta_host, cm_meta_host);
}
