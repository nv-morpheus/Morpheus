/**
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

#include "morpheus/io/deserializers.hpp"

#include "morpheus/utilities/cudf_util.hpp"  // for proxy_table_from_table_with_metadata
#include "morpheus/utilities/stage_util.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/scalar/scalar.hpp>  // for string_scalar
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>  // IWYU pragma: keep
#include <cudf/types.hpp>        // for cudf::type_id
#include <ext/alloc_traits.h>
#include <glog/logging.h>
#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ostream>  // needed for logging
#include <regex>
#include <utility>

namespace {
const std::regex IndexRegex(R"(^\s*(unnamed: 0|id)\s*$)",
                            std::regex_constants::ECMAScript | std::regex_constants::icase);

const std::regex UnnamedRegex(R"(^\s*unnamed: 0\s*$)", std::regex_constants::ECMAScript | std::regex_constants::icase);
}  // namespace

namespace morpheus {

std::vector<std::string> get_column_names_from_table(const cudf::io::table_with_metadata& table)
{
    DCHECK(!(!table.metadata.column_names.empty() && !table.metadata.schema_info.empty()))
        << "Both column_names and schema_info were set on the table_with_metadata object. Defaulting to column_names";

    // If column_names is populated, use that
    if (!table.metadata.column_names.empty())
    {
        return table.metadata.column_names;
    }

    // Otherwise, use schema_info
    if (!table.metadata.schema_info.empty())
    {
        return foreach_map(table.metadata.schema_info, [](auto schema) { return schema.name; });
    }

    // Return empty
    return {};
}

cudf::io::table_with_metadata load_json_table(cudf::io::json_reader_options&& json_options)
{
    auto tbl = cudf::io::read_json(json_options);

    auto column_names = get_column_names_from_table(tbl);

    auto found = std::find(column_names.begin(), column_names.end(), "data");

    if (found == column_names.end())
        return tbl;

    // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
    // \\n -> \n
    // \\/ -> \/
    auto columns = tbl.tbl->release();

    size_t idx = found - column_names.begin();

    auto updated_data = cudf::strings::replace(
        cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"), cudf::string_scalar("\n"));

    updated_data = cudf::strings::replace(
        cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"), cudf::string_scalar("/"));

    columns[idx] = std::move(updated_data);

    tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

    return tbl;
}

cudf::io::table_with_metadata load_table_from_file(const std::string& filename, FileTypes file_type)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    if (file_type == FileTypes::JSON)
    {
        // First, load the file into json
        auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(true);
        return load_json_table(options.build());
    }
    else  // CSV
    {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});
        return cudf::io::read_csv(options.build());
    }
}

pybind11::object read_file_to_df(const std::string& filename, FileTypes file_type)
{
    auto table          = load_table_from_file(filename, file_type);
    int index_col_count = prepare_df_index(table);

    pybind11::gil_scoped_acquire gil;
    return proxy_table_from_table_with_metadata(std::move(table), index_col_count);
}

int get_index_col_count(const cudf::io::table_with_metadata& data_table)
{
    int index_col_count   = 0;
    const auto& col_names = data_table.metadata.column_names;

    // Check if we have a first column with INT64 data type
    if (col_names.size() >= 1 && data_table.tbl->get_column(0).type().id() == cudf::type_id::INT64)
    {
        // Get the column name
        const auto& col_name = col_names[0];

        // Check it against some common terms
        if (std::regex_search(col_name, IndexRegex))
        {
            index_col_count = 1;
        }
    }

    return index_col_count;
}

int prepare_df_index(cudf::io::table_with_metadata& data_table)
{
    const int index_col_count = get_index_col_count(data_table);

    if (index_col_count > 0)
    {
        auto& col_names = data_table.metadata.column_names;
        auto& col_name  = col_names[0];

        // Also, if its the hideous 'Unnamed: 0', then just use an empty string
        if (std::regex_search(col_name, UnnamedRegex))
        {
            col_name.clear();
        }
    }

    return index_col_count;
}

}  // namespace morpheus
