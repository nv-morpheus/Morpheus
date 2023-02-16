/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/messages/multi_inference.hpp"
#include "morpheus/messages/multi_response.hpp"
#include "morpheus/objects/dtype.hpp"  // for TypeId
#include "morpheus/objects/tensor.hpp"

#include <cuda_runtime.h>  // for cudaMemcpy, cudaMemcpyHostToDevice
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <pybind11/embed.h>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>  // for unique_ptr
#include <random>
#include <string>
#include <typeinfo>  //for typeid
#include <vector>

using namespace morpheus;

TEST_CLASS(Deserializers);

TEST_F(TestDeserializers, GetIndexColCount)
{
    std::filesystem::path morpheus_root{std::getenv("MORPHEUS_ROOT")};
    auto test_data_dir = morpheus_root / "tests/tests_data";

    {
        // First test a files without an index
        std::vector<std::filesystem::path> input_files{test_data_dir / "filter_probs.csv",
                                                       test_data_dir / "filter_probs.jsonlines"};
        for (const auto& input_file : input_files)
        {
            auto table = load_table_from_file(input_file);
            EXPECT_EQ(get_index_col_count(table), 0);
        }
    }

    {
        // now test a file with an index
        auto input_file = morpheus_root / "tests/tests_data/filter_probs_w_id_col.csv";
        auto table      = load_table_from_file(input_file);
        EXPECT_EQ(get_index_col_count(table), 1);
    }
}
