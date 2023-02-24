/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "morpheus/io/serializers.hpp"
#include "morpheus/messages/meta.hpp"

#include <cudf/io/types.hpp>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>  // for stringstream
#include <string>
#include <vector>

using namespace morpheus;

std::string read_file(const std::filesystem::path& file_path)
{
    std::fstream in_stream{file_path, in_stream.in};
    std::stringstream buff;
    in_stream >> buff.rdbuf();
    return buff.str();
}

class TestFileInOut : public morpheus::test::TestWithPythonInterpreter
{};

TEST_F(TestFileInOut, RoundTripCSV)
{
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    std::vector<std::filesystem::path> input_files{test_data_dir / "filter_probs.csv",
                                                   test_data_dir / "filter_probs_w_id_col.csv"};
    for (const auto& input_file : input_files)
    {
        auto table           = load_table_from_file(input_file);
        auto index_col_count = prepare_df_index(table);

        /*
        auto meta            = MessageMeta::create_from_cpp(std::move(table), index_col_count);

        const auto csv_data = df_to_csv(meta->get_info(), true, index_col_count > 0);
        const auto src_data = read_file(input_file);

        EXPECT_EQ(csv_data, src_data);
        */
    }
}

TEST_F(TestFileInOut, RoundTripJSON)
{
    using nlohmann::json;
    auto input_file      = test::get_morpheus_root() / "tests/tests_data/filter_probs.jsonlines";
    auto table           = load_table_from_file(input_file);
    auto index_col_count = prepare_df_index(table);

    EXPECT_EQ(index_col_count, 0);

    /*
    auto meta         = MessageMeta::create_from_cpp(std::move(table), index_col_count);
    auto mutable_info = meta->get_mutable_info();

    // Two JSON strings might be equivelant even if the strings are not. ("{\"a\": 5}"" != "{\"a\":5}")
    const auto json_str = df_to_json(mutable_info);
    const auto src_str  = read_file(input_file);

    const auto json_data = json::parse(json_str);
    const auto src_data  = json::parse(src_str);

    EXPECT_EQ(json_data, src_data);
    */
}
