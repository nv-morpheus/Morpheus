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

#include "../test_utils/common.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi.hpp"
#include "morpheus/stages/add_classification.hpp"
#include "morpheus/types.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>

using namespace morpheus;

TEST_CLASS(AddClassification);

TEST_F(TestAddClassification, TestProcessControlMessageAndMultiResponseMessage)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "bools.csv";

    TensorIndex cols_size  = 3;
    TensorIndex mess_count = 3;
    auto packed_data =
        std::make_shared<rmm::device_buffer>(cols_size * mess_count * sizeof(double), rmm::cuda_stream_per_thread);

    cudf::io::csv_reader_options read_opts = cudf::io::csv_reader_options::builder(cudf::io::source_info(input_file))
                                                 .dtypes({cudf::data_type(cudf::data_type{cudf::type_to_id<bool>()})})
                                                 .header(0);
    cudf::io::table_with_metadata table_with_meta = cudf::io::read_csv(read_opts);
    auto meta                                     = MessageMeta::create_from_cpp(std::move(table_with_meta));

    std::map<std::size_t, std::string> idx2label = {{0, "bool"}};

    // Create MultiResponseMessage
    auto tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto tensor_memory = std::make_shared<TensorMemory>(mess_count);
    tensor_memory->set_tensor("probs", std::move(tensor));
    auto mm = std::make_shared<MultiResponseMessage>(meta, 0, mess_count, std::move(tensor_memory));

    // Create PreProcessMultiMessageStage
    auto mm_stage =
        std::make_shared<AddClassificationsStage<MultiResponseMessage, MultiResponseMessage>>(idx2label, 0.0);
    auto mm_response = mm_stage->on_data(mm);

    // Create ControlMessage
    auto cm = std::make_shared<ControlMessage>();
    cm->payload(meta);
    auto cm_tensor        = Tensor::create(packed_data, DType::create<double>(), {mess_count, cols_size}, {}, 0);
    auto cm_tensor_memory = std::make_shared<TensorMemory>(mess_count);
    cm_tensor_memory->set_tensor("probs", std::move(cm_tensor));
    cm->tensors(cm_tensor_memory);

    // Create PreProcessControlMessageStage
    auto cm_stage    = std::make_shared<AddClassificationsStage<ControlMessage, ControlMessage>>(idx2label, 0.0);
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
