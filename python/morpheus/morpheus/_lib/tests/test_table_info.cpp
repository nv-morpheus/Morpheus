/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/io/deserializers.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/table_info.hpp"  // for TableInfo

#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/table/table_view.hpp>    // for table_view
#include <cudf/types.hpp>               // for data_type
#include <gtest/gtest.h>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <cstddef>
#include <filesystem>
#include <fstream>  // IWYU pragma: keep
#include <memory>   // for shared_ptr
#include <string>
#include <utility>  // for move
#include <vector>
// IWYU pragma: no_include <initializer_list>

using namespace morpheus;

namespace {
TableInfo table_info_from_file(const std::filesystem::path& input_file)
{
    auto table           = load_table_from_file(input_file);
    auto index_col_count = prepare_df_index(table);

    auto meta = MessageMeta::create_from_cpp(std::move(table), index_col_count);

    pybind11::gil_scoped_release no_gil;
    return meta->get_info();
}

bool column_view_is_same(const cudf::column_view& cv1, const cudf::column_view& cv2)
{
    // We want to verify that two column_views are the same (not holding equivelant values, but are views to the same
    // column), we will consider two column_views to be the same if:
    // 1. The number of rows is the same
    // 2. The data types are the same
    // 3. The head pointer is the same
    // 4. The offset is the same
    return (cv1.size() == cv2.size()) && (cv1.type() == cv2.type()) && (cv1.head() == cv2.head()) &&
           (cv1.offset() == cv2.offset());
}
}  // namespace

class TestTableInfo : public morpheus::test::TestWithPythonInterpreter
{};

TEST_F(TestTableInfo, GetColumn)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file{test_data_dir / "filter_probs.csv"};

    auto table_info = table_info_from_file(input_file);
    auto table_view = table_info.get_view();

    auto column_names = table_info.get_column_names();
    for (std::size_t i = 0; i < column_names.size(); ++i)
    {
        auto cv          = table_info.get_column(i);
        auto expected_cv = table_view.column(i + 1);  // +1 because the first column is the index

        EXPECT_TRUE(column_view_is_same(cv, expected_cv))
            << "Column " << i << " (" << column_names[i] << ") does not match expected column.";
    }
}

TEST_F(TestTableInfo, GetColumnByName)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file{test_data_dir / "filter_probs.csv"};

    auto table_info = table_info_from_file(input_file);
    auto table_view = table_info.get_view();

    auto column_names = table_info.get_column_names();
    for (std::size_t i = 0; i < column_names.size(); ++i)
    {
        auto cv          = table_info.get_column(column_names[i]);
        auto expected_cv = table_view.column(i + 1);  // +1 because the first column is the index

        EXPECT_TRUE(column_view_is_same(cv, expected_cv))
            << "Column " << i << " (" << column_names[i] << ") does not match expected column.";
    }
}
