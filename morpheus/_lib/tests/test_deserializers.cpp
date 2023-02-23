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

#include <cudf/io/types.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

using namespace morpheus;

TEST_CLASS(Deserializers);

TEST_F(TestDeserializers, GetIndexColCountNoIdx)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    // First test a files without an index
    std::vector<std::filesystem::path> input_files{test_data_dir / "filter_probs.csv",
                                                   test_data_dir / "filter_probs.jsonlines"};
    for (const auto& input_file : input_files)
    {
        auto table = load_table_from_file(input_file);
        EXPECT_EQ(get_index_col_count(table), 0);
    }
}

TEST_F(TestDeserializers, GetIndexColCountWithIdx)
{
    // now test a file with an index
    auto input_file = test::get_morpheus_root() / "tests/tests_data/filter_probs_w_id_col.csv";
    auto table      = load_table_from_file(input_file);
    EXPECT_EQ(get_index_col_count(table), 1);
}
