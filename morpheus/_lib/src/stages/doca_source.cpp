/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/doca_source.hpp"

#include <cudf/column/column.hpp>  // for column
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>  // for strings_column_view
#include <cudf/table/table.hpp>                  // for table
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/cast.h>  // for object_api::operator()
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_
#include <srf/segment/builder.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>  // for find
#include <cstddef>    // for size_t
#include <filesystem>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>  // for runtime_error
#include <utility>
// IWYU thinks we need __alloc_traits<>::value_type for vector assignments
// IWYU pragma: no_include <ext/alloc_traits.h>

namespace morpheus {
// Component public implementations
// ************ DocaSourceStage ************* //
DocaSourceStage::DocaSourceStage() :
  PythonSource(build())
{
  _context = std::make_shared<morpheus::doca::doca_context>("a3:00.0", "a6:00.0");
}

DocaSourceStage::subscriber_fn_t DocaSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> output) {

        auto stream = cudf::default_stream_value;
        auto values = rmm::device_uvector<int32_t>(1000000, stream);

        // thrust::uninitialized_fill(thrust::cuda::par.on(stream.value()), values.begin(), values.end(), int32_t{0});

        auto values_size = values.size();
        auto my_column   = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_to_id<int32_t>()},
          values_size,
          values.release());

        auto my_columns          = std::vector<std::unique_ptr<cudf::column>>();

        my_columns.push_back(std::move(my_column));

        // auto my_table            = std::make_unique<cudf::table>(std::move(my_columns));
        auto metadata            = cudf::io::table_metadata();

        metadata.column_names.push_back("index");
        
        auto my_table_w_metadata = cudf::io::table_with_metadata{
          std::make_unique<cudf::table>(std::move(my_columns)),
          std::move(metadata)
        };
        auto meta                = MessageMeta::create_from_cpp(std::move(my_table_w_metadata), 1);

        output.on_next(meta);

        output.on_completed();
    };
}

// cudf::io::table_with_metadata DocaSourceStage::load_table()
// {
//     auto file_path = std::filesystem::path(m_filename);

//     if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
//     {
//         // First, load the file into json
//         auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{m_filename}).lines(true);

//         auto tbl = cudf::io::read_json(options.build());

//         auto found = std::find(tbl.metadata.column_names.begin(), tbl.metadata.column_names.end(), "data");

//         if (found == tbl.metadata.column_names.end())
//             return tbl;

//         // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
//         // \\n -> \n
//         // \\/ -> \/
//         auto columns = tbl.tbl->release();

//         size_t idx = found - tbl.metadata.column_names.begin();

//         auto updated_data = cudf::strings::replace(
//             cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"), cudf::string_scalar("\n"));

//         updated_data = cudf::strings::replace(
//             cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"), cudf::string_scalar("/"));

//         columns[idx] = std::move(updated_data);

//         tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

//         return tbl;
//     }
//     else if (file_path.extension() == ".csv")
//     {
//         auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{m_filename});

//         return cudf::io::read_csv(options.build());
//     }
//     else
//     {
//         LOG(FATAL) << "Unknown extension for file: " << m_filename;
//         throw std::runtime_error("Unknown extension");
//     }
// }

// ************ DocaSourceStageInterfaceProxy ************ //
std::shared_ptr<srf::segment::Object<DocaSourceStage>> DocaSourceStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name)
{
    auto stage = builder.construct_object<DocaSourceStage>(name);

    return stage;
}
}  // namespace morpheus
