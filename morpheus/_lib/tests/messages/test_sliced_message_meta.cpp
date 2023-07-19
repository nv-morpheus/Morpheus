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

#include "../test_morpheus.hpp"  // IWYU pragma: associated

#include "morpheus/io/deserializers.hpp"     // for load_table_from_file, prepare_df_index
#include "morpheus/messages/meta.hpp"        // for MessageMeta and SlicedMessageMeta
#include "morpheus/messages/multi.hpp"       // for MultiMessage
#include "morpheus/objects/table_info.hpp"   // for TableInfo
#include "morpheus/utilities/cudf_util.hpp"  // for CudfHelper

#include <gtest/gtest.h>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <filesystem>  // for std::filesystem::path
#include <memory>      // for shared_ptr
#include <utility>     // for move

using namespace morpheus;
using namespace morpheus::test;

class TestSlicedMessageMeta : public morpheus::test::TestWithPythonInterpreter
{
  protected:
    void SetUp() override
    {
        morpheus::test::TestWithPythonInterpreter::SetUp();
        {
            pybind11::gil_scoped_acquire gil;

            // Initially I ran into an issue bootstrapping cudf, I was able to work-around the issue, details in:
            // https://github.com/rapidsai/cudf/issues/12862
            CudfHelper::load();
        }
    }
};

TEST_F(TestSlicedMessageMeta, TestCount)
{
    // Test for issue #970
    auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";

    auto input_file{test_data_dir / "filter_probs.csv"};

    auto table           = load_table_from_file(input_file);
    auto index_col_count = prepare_df_index(table);

    auto meta = MessageMeta::create_from_cpp(std::move(table), index_col_count);
    EXPECT_EQ(meta->count(), 20);

    SlicedMessageMeta sliced_meta(meta, 5, 15);
    EXPECT_EQ(sliced_meta.count(), 10);

    // Ensure the count value matches the table info
    pybind11::gil_scoped_release no_gil;
    EXPECT_EQ(sliced_meta.get_info().num_rows(), sliced_meta.count());

    // ensure count is correct when using a pointer to the parent class which is the way Python will use it
    auto p_sliced_meta = std::make_shared<SlicedMessageMeta>(meta, 5, 15);
    auto p_meta        = std::dynamic_pointer_cast<MessageMeta>(p_sliced_meta);
    EXPECT_EQ(p_meta->count(), 10);
    EXPECT_EQ(p_meta->get_info().num_rows(), p_meta->count());
}

TEST_F(TestSlicedMessageMeta, TestDocEx3)
{
    // Test to verify that the lambda in docs/source/developer_guide/guides/3_simple_cpp_stage.md
    // compiles and runs correctly
    using namespace pybind11::literals;
    pybind11::gil_scoped_release no_gil;

    auto doc_fn = [](std::shared_ptr<MultiMessage> msg) {
        auto mutable_info = msg->meta->get_mutable_info();

        std::shared_ptr<MessageMeta> new_meta;
        {
            pybind11::gil_scoped_acquire gil;
            auto df = mutable_info.checkout_obj();

            // Maka a copy of the original DataFrame
            auto copied_df = df.attr("copy")("deep"_a = true);

            // Now that we are done with `df` return it to the owner
            mutable_info.return_obj(std::move(df));

            // do something with copied_df
            new_meta = MessageMeta::create_from_python(std::move(copied_df));
        }  // GIL is released

        return new_meta;
    };

    auto msg_meta = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);
    auto msg      = std::make_shared<MultiMessage>(msg_meta);

    auto result = doc_fn(msg);

    auto orig_mutable_info   = msg_meta->get_mutable_info();
    auto result_mutable_info = result->get_mutable_info();
    {
        pybind11::gil_scoped_acquire gil;

        auto orig_df   = orig_mutable_info.checkout_obj();
        auto result_df = result_mutable_info.checkout_obj();

        // orig_df.eq(result_df).all().all()
        auto is_true = orig_df.attr("eq")(result_df).attr("all")().attr("all")();
        EXPECT_TRUE(is_true.cast<bool>());

        orig_mutable_info.return_obj(std::move(orig_df));
        result_mutable_info.return_obj(std::move(result_df));
    }
}
