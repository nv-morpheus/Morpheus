/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_morpheus.hpp"  // IWYU pragma: associated

#include "morpheus/utilities/matx_util.hpp"
#include "morpheus/utilities/type_util.hpp"
#include "morpheus/utilities/type_util_detail.hpp"

#include <cudf/table/table.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <srf/cuda/common.hpp>

#include <cstdlib>  // for std::getenv
#include <filesystem>
#include <string>
#include <vector>

using namespace morpheus;

TEST_CLASS(MatxUtil);

TEST_F(TestMatxUtil, ReduceMax)
{
    // Test mimics example from the method's docstring
    std::vector<float> input{5, 2, 8, 9, 8, 2, 1};
    std::vector<int32_t> seq_ids{0, 0, 0, 1, 2, 3, 3};
    std::vector<float> expected_output{8, 9, 8, 2};

    DataType dtype(TypeId::FLOAT32);

    auto input_buffer =
        std::make_shared<rmm::device_buffer>(input.size() * dtype.item_size(), rmm::cuda_stream_per_thread);

    SRF_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input.size(), dtype.type_id(), input_buffer, 0};
    std::vector<int64_t> input_shape{static_cast<int64_t>(input.size()), 1};
    std::vector<int64_t> output_shape{static_cast<int64_t>(expected_output.size()), 1};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, input_shape, output_shape);

    std::vector<float> output(expected_output.size());
    SRF_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output, expected_output);
}

TEST_F(TestMatxUtil, ReduceMax2d)
{
    std::filesystem::path morpheus_root{std::getenv("MORPHEUS_ROOT")};
    auto input_file = morpheus_root / "tests/tests_data/filter_probs.csv";

    auto table_m  = load_table_from_csv(input_file);
    auto num_rows = table_m.tbl->num_rows();
    auto num_cols = table_m.tbl->num_columns();
    EXPECT_EQ(num_rows, 20);
    EXPECT_EQ(num_cols, 4);

    // Copy data from table into one big buffer
    std::size_t buff_size{0};
    std::vector<std::unique_ptr<rmm::device_buffer>> column_data;
    auto columns = table_m.tbl->release();
    auto dtype   = DType::from_cudf(columns[0]->type().id());
    for (const auto& c : columns)
    {
        auto contents = c->release();
        buff_size += contents.data->size();
        column_data.emplace_back(std::move(contents.data));
    }

    auto input_buffer = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    std::size_t offset{0};
    for (const auto& c : column_data)
    {
        SRF_CHECK_CUDA(cudaMemcpy(
            static_cast<uint8_t*>(input_buffer->data()) + offset, c->data(), c->size(), cudaMemcpyDeviceToDevice));

        offset += c->size();
    }

    // reducing 20 rows down to 12
    std::vector<int32_t> seq_ids{0, 0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 7, 8, 9, 9, 10, 11};
    // disabling formatting so I can enter the literal values by column
    // clang-format off
    std::vector<float> expected_output{0.1, 1.0, 1.0, 1.0, 0.5, 0.3, 0.9, 0.5, 0.0, 0.6, 0.8, 0.1,
                                       0.7, 0.9, 0.6, 0.2, 0.8, 0.4, 0.3, 0.5, 0.3, 1.0, 0.8, 0.9,
                                       0.7, 0.5, 0.7, 0.2, 0.6, 0.1, 1.0, 0.6, 0.5, 0.8, 1.0, 0.1,
                                       0.7, 0.9, 0.9, 0.9, 0.0, 0.4, 0.6, 0.8, 0.6, 0.7, 0.6, 0.3};
    // clang-format on
    const std::size_t expected_rows = expected_output.size();

    DevMemInfo dm{static_cast<std::size_t>(num_rows * num_cols), dtype.type_id(), input_buffer, 0};
    std::vector<int64_t> input_shape{static_cast<int64_t>(num_rows), static_cast<int64_t>(num_cols)};
    std::vector<int64_t> output_shape{static_cast<int64_t>(expected_rows), static_cast<int64_t>(num_cols)};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, input_shape, output_shape);

    EXPECT_EQ(output_buffer->size(), expected_rows * num_cols * dtype.item_size());

    std::vector<float> output(expected_rows * num_cols);
    SRF_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output, expected_output);
}
