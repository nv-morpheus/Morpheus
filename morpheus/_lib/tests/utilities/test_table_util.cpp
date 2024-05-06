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

#include "morpheus/io/deserializers.hpp"
#include "morpheus/utilities/table_util.hpp"  // for filter_null_data

#include <cudf/stream_compaction.hpp>  // for drop_nulls
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>
// IWYU pragma: no_include <initializer_list>

using namespace morpheus;

TEST_CLASS(TableUtil);

TEST_F(TestTableUtil, Null_Data)
{
    auto morpheus_root = test::get_morpheus_root();
    auto input_files   = {morpheus_root / "tests/tests_data/file_with_nans.csv",
                          morpheus_root / "tests/tests_data/file_with_nans.jsonlines",
                          morpheus_root / "tests/tests_data/file_with_nulls.csv",
                          morpheus_root / "tests/tests_data/file_with_nulls.jsonlines"};

    for (const auto& input_file : input_files)
    {
        auto table_w_meta    = load_table_from_file(input_file);
        auto index_col_count = prepare_df_index(table_w_meta);

        EXPECT_EQ(index_col_count, 0);

        EXPECT_EQ(table_w_meta.tbl->num_columns(), 2);
        EXPECT_EQ(table_w_meta.tbl->num_rows(), 10);

        std::vector<std::string> filter_columns{"data"};
        CuDFTableUtil::filter_null_data(table_w_meta, filter_columns);

        EXPECT_EQ(table_w_meta.tbl->num_columns(), 2);
        EXPECT_EQ(table_w_meta.tbl->num_rows(), 8);
    }
}
