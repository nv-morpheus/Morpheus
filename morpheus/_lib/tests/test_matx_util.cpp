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

#include <string>
#include <vector>

using namespace morpheus;

TEST_CLASS(MatxUtil);

TEST_F(TestMatxUtil, ReduceMax1d)
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

TEST_F(TestMatxUtil, ReduceMax2dRowMajor)
{
    // clang-format off
    // disabling clang-format to illustrate row-major layout
    std::vector<double> input{
        0.1, 0.7, 0.7, 0.7,
        1.0, 0.9, 0.5, 0.9,
        1.0, 0.6, 0.7, 0.9,
        1.0, 0.2, 0.2, 0.9,
        0.5, 0.8, 0.6, 0.0,
        0.3, 0.4, 0.1, 0.4,
        0.9, 0.3, 1.0, 0.6,
        0.5, 0.5, 0.6, 0.8,
        0.0, 0.3, 0.5, 0.6,
        0.6, 1.0, 0.8, 0.7,
        0.8, 0.8, 1.0, 0.6,
        0.1, 0.9, 0.1, 0.3};

    // reducing 12 rows down to 5
    std::vector<int32_t> seq_ids{0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4};

    std::vector<double> expected_output{
        1.0, 0.9, 0.7, 0.9,
        1.0, 0.6, 0.7, 0.9,
        1.0, 0.8, 1.0, 0.9,
        0.8, 1.0, 1.0, 0.8,
        0.1, 0.9, 0.1, 0.3};
    // clang-format on

    // Copy data from table into one big buffer
    std::size_t num_cols      = 4;
    std::size_t num_rows      = 12;
    std::size_t expected_rows = expected_output.size() / num_cols;

    EXPECT_EQ(num_cols * num_rows, input.size());
    EXPECT_EQ(expected_rows, 5);
    DataType dtype(TypeId::FLOAT64);
    EXPECT_EQ(dtype.item_size(), sizeof(double));

    std::size_t buff_size = input.size() * dtype.item_size();
    auto input_buffer     = std::make_shared<rmm::device_buffer>(buff_size, rmm::cuda_stream_per_thread);

    SRF_CHECK_CUDA(cudaMemcpy(input_buffer->data(), input.data(), input_buffer->size(), cudaMemcpyHostToDevice));

    DevMemInfo dm{input.size(), dtype.type_id(), input_buffer, 0};
    std::vector<int64_t> input_shape{static_cast<int64_t>(num_rows), static_cast<int64_t>(num_cols)};
    std::vector<int64_t> output_shape{static_cast<int64_t>(expected_rows), static_cast<int64_t>(num_cols)};
    auto output_buffer = MatxUtil::reduce_max(dm, seq_ids, 0, input_shape, output_shape);

    EXPECT_EQ(output_buffer->size(), expected_rows * num_cols * dtype.item_size());

    std::vector<double> output(expected_rows * num_cols);
    SRF_CHECK_CUDA(cudaMemcpy(output.data(), output_buffer->data(), output_buffer->size(), cudaMemcpyDeviceToHost));

    EXPECT_EQ(output.size(), expected_output.size());
    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(output[i], expected_output[i]);
    }
}
