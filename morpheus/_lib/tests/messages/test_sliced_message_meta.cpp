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

#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/io/deserializers.hpp"     // for load_table_from_file, prepare_df_index
#include "morpheus/messages/meta.hpp"        // for MessageMeta and SlicedMessageMeta
#include "morpheus/objects/python_data_table.hpp"
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
    std::cerr << meta->count();

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

// TEST_F(TestSlicedMessageMeta, TestMeta2)
// {
//     // auto mm = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    

//     auto string_df = create_mock_csv_file({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

//     // auto string_df = "[{\"c0\":{\"k0\":\"v0\",\"k1\":\"v3\"},\"c1\":0},{\"c0\":{\"k0\":\"v1\",\"k1\":\"v4\"},\"c1\":1},{\"c0\":{\"k0\":\"v2\",\"k1\":\"v5\"},\"c1\":2}]";
//     // auto string_df = "{\"c0\":{\"k0\":\"v0\",\"k1\":\"v3\"},\"c1\":0}\n{\"c0\":{\"k0\":\"v1\",\"k1\":\"v4\"},\"c1\":1}\n{\"c0\":{\"k0\":\"v2\",\"k1\":\"v5\"},\"c1\":2}]";

//     // pybind11::gil_scoped_release no_gil;
//     pybind11::gil_scoped_acquire gil;
//     pybind11::module_ mod_cudf;
//     mod_cudf = pybind11::module_::import("cudf");

//     auto py_string = pybind11::str(string_df);
//     auto py_buffer = pybind11::buffer(pybind11::bytes(py_string));
//     auto dataframe = mod_cudf.attr("read_csv")(py_buffer);
//     // auto dataframe = mod_cudf.attr("read_json")(py_buffer);


//     // auto options =
//     //     cudf::io::json_reader_options::builder(cudf::io::source_info(py_string)).lines(true);

//     // auto dataframe = cudf::io::read_json(options.build());
//     // auto dataframe = mod_cudf.attr("read_json")(pybind11::str(string_df), pybind11::str("cudf"));

//     // auto data = std::make_unique<PyDataTable>(std::move(dataframe));

//     auto meta = MessageMeta::create_from_python(std::move(dataframe));

//     pybind11::gil_scoped_release no_gil;

//     auto info = meta->get_info();

//     auto view = info.get_view();

//     auto col = view.column(0);

    

//     // auto data = std::make_unique<PyDataTable>(std::move(dataframe));


//     // Get the column and convert to cudf
//     // TableInfo info = mm->get_info();

//     // auto table_info_data = info.get_data();

//     // auto ret = morpheus::CudfHelper::table_from_table_info(info);

//     std::cerr << "hello";
// }

// TEST_F(TestSlicedMessageMeta, TestNested)
// {
//     auto test_data_dir = test::get_morpheus_root() / "tests/tests_data";
    
//     // read json file to dataframe
//     auto input_file{test_data_dir / "nested.json"};
//     auto source_info = cudf::io::source_info(input_file);
//     auto input_builder     = cudf::io::json_reader_options::builder(source_info).lines(true);
//     auto input_options     = input_builder.build();
//     // auto [input, metadata] =  cudf::io::read_json(input_options);
    

//     // pybind11::gil_scoped_acquire gil;
//     auto table =  cudf::io::read_json(input_options);
//     // auto metadata = table.metadata;
//     auto index_col_count = prepare_df_index(table);
//     auto meta = MessageMeta::create_from_cpp(std::move(table), index_col_count);

//     TableInfo table_info;

//     {
//         pybind11::gil_scoped_release no_gil;
//         table_info = meta->get_info();
//     }
//     // auto cudf_table = CudfHelper::table_from_table_info(info);

//     std::cerr << "hello";

//     // write dataframe to json file

//     auto output_file{test_data_dir / "nested_out.json"};
//     auto sink_info = cudf::io::sink_info(output_file);
//     // auto output_builder   = cudf::io::json_writer_options::builder(sink_info, input->view()).lines(true);
//     // output_builder.metadata(metadata);

//     // auto output_builder   = cudf::io::json_writer_options::builder(sink_info, info.get_view()).lines(true);
//     // // output_builder.metadata(metadata);
//     // auto output_options = output_builder.build();
//     // cudf::io::write_json(output_options);

//     // auto output_builder   = cudf::io::json_writer_options::builder(sink_info, table.tbl->view()).lines(true);
//     // output_builder.metadata(table.metadata);
//     // auto output_options = output_builder.build();
//     // cudf::io::write_json(output_options);
// }