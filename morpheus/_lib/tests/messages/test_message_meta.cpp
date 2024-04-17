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
#include "../test_utils/tensor_utils.hpp"
#include "test_messages.hpp"

#include "morpheus/io/deserializers.hpp"  // for load_table_from_file, prepare_df_index
#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"  // for MessageMeta and SlicedMessageMeta
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/rmm_tensor.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo
#include "morpheus/objects/tensor.hpp"
#include "morpheus/stages/preallocate.hpp"
#include "morpheus/utilities/cudf_util.hpp"  // for CudfHelper

#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <filesystem>  // for std::filesystem::path
#include <memory>      // for shared_ptr
#include <utility>     // for move

using namespace morpheus;
using namespace morpheus::test;

using TestMessageMeta = morpheus::test::TestMessages;  // NOLINT(readability-identifier-naming)

TEST_F(TestMessageMeta, SetdataWithColumnName)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "csv_sample.csv";

    auto meta  = MessageMeta::create_from_cpp(load_table_from_file(input_file));

    std::vector<int64_t> packed_data_host{9, 8, 7, 6, 5, 4, 3, 2, 1};
    int mess_count = packed_data_host.size();
    int cols_size = 1;
    auto packed_data = std::make_shared<rmm::device_buffer>(packed_data_host.data(), mess_count * cols_size * sizeof(int64_t), rmm::cuda_stream_per_thread);

    auto tensor = Tensor::create(packed_data, DType::create<int64_t>(), {mess_count, cols_size}, {}, 0);
    meta->set_data("int", tensor);

    assert_eq_device_to_host(meta->get_info().get_column(0), packed_data_host);
}
