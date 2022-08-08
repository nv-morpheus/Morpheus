/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cstdlib>
#include <filesystem>
#include <vector>

TEST_CLASS(Masking);

TEST_F(TestMasking, Ranges)
{
    std::filesystem::path morpheus_root{std::getenv("MORPHEUS_ROOT")};
    auto input_file = morpheus_root / "tests/tests_data/filter_probs.csv";

    auto table_m = load_table_from_csv(input_file);
    EXPECT_EQ(table_m.tbl->num_rows(), 20);

    auto table_v = table_m.tbl->view();
    EXPECT_EQ(table_v.num_rows(), table_m.tbl->num_rows());

    std::vector<cudf::size_type> ranges{2, 4, 12, 14};
    auto slices = cudf::slice(table_v, ranges);

    // We should get two slices both with 2 rows
    EXPECT_EQ(slices.size(), 2);
    for (const auto& s : slices)
    {
        EXPECT_EQ(s.num_rows(), 2);
    }

    auto table_c = cudf::concatenate(slices);
    EXPECT_EQ(table_c->num_rows(), 4);
}
