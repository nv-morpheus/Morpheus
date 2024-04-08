/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // IWYU pragma: associated
#include "test_messages.hpp"

#include "morpheus/io/deserializers.hpp"  // for load_table_from_file, prepare_df_index
#include "morpheus/messages/meta.hpp"     // for MessageMeta and SlicedMessageMeta
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/table_info.hpp"   // for TableInfo
#include "morpheus/utilities/cudf_util.hpp"  // for CudfHelper

#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <filesystem>  // for std::filesystem::path
#include <memory>      // for shared_ptr
#include <utility>     // for move

using namespace morpheus;

using TestMessageMeta = morpheus::test::TestMessages;  // NOLINT(readability-identifier-naming)

TEST_F(TestMessageMeta, SetMetaWithColumnName)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "csv_sample.csv";

    auto table = load_table_from_file(input_file);
    auto meta  = MessageMeta::create_from_cpp(std::move(table));

    const std::size_t count = 3;
    DType int_type(TypeId::INT64);
    std::vector<int64_t> expected_ints{4, 5, 6};
    auto buffer = std::make_shared<rmm::device_buffer>(count * int_type.item_size(), rmm::cuda_stream_per_thread);

    MRC_CHECK_CUDA(cudaMemcpy(buffer->data(), expected_ints.data(), buffer->size(), cudaMemcpyHostToDevice));

    ShapeType shape{3, 1};
    auto tensor = std::make_shared<RMMTensor>(buffer, 0, int_type, shape);
    TensorObject tensor_object(tensor);
    meta->set_data("int", tensor_object);

    std::vector<int64_t> actual_ints(expected_ints.size());

    auto cm_int_meta = meta->get_info().get_column(0);
    MRC_CHECK_CUDA(
        cudaMemcpy(actual_ints.data(), cm_int_meta.data<int64_t>(), count * sizeof(int64_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(expected_ints, actual_ints);
}
