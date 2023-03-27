/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/table_info_data.hpp"

// #include "morpheus/objects/dtype.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <utility>

// #include <algorithm>  // for find, transform
// #include <array>      // needed for pybind11::make_tuple
// #include <cstddef>    // for size_t
// #include <iterator>   // for back_insert_iterator, back_inserter
// #include <memory>
// #include <optional>
// #include <ostream>
// #include <shared_mutex>
// #include <stdexcept>
// #include <utility>
// IWYU pragma: no_include <pybind11/cast.h>

namespace morpheus {

TableInfoData::TableInfoData(cudf::table_view view,
                             std::vector<std::string> indices,
                             std::vector<std::string> columns) :
  table_view(std::move(view)),
  index_names(std::move(indices)),
  column_names(std::move(columns))
{}

TableInfoData TableInfoData::get_slice(std::vector<std::string> column_names) const
{
    return this->get_slice(0, -1, std::move(column_names));
}

TableInfoData TableInfoData::get_slice(cudf::size_type start,
                                       cudf::size_type stop,
                                       std::vector<std::string> column_names) const
{
    CHECK_GE(start, 0) << "Start must be >= 0";

    if (stop < 0)
    {
        stop = this->table_view.num_rows();
    }

    CHECK_GT(stop, 0) << "Stop must be > 0";
    CHECK_LE(stop, this->table_view.num_rows()) << "Stop must be less than the number of rows";
    CHECK_LE(start, stop) << "Start must be less than stop";

    if (column_names.empty())
    {
        column_names = this->column_names;
    }

    // Start with our table view
    auto table_view_out = this->table_view;

    // If the columns are different, calculate the new slice
    if (column_names != this->column_names)
    {
        std::vector<cudf::size_type> col_indices;

        std::vector<std::string> new_column_names;

        // Append the indices column idx by default
        for (cudf::size_type i = 0; i < this->index_names.size(); ++i)
        {
            col_indices.push_back(i);
        }

        std::transform(column_names.begin(),
                       column_names.end(),
                       std::back_inserter(col_indices),
                       [this, &new_column_names](const std::string& c) {
                           auto found_col = std::find(this->column_names.begin(), this->column_names.end(), c);

                           if (found_col == this->column_names.end())
                           {
                               throw std::invalid_argument("Unknown column: " + c);
                           }

                           // Add the found column to the metadata
                           new_column_names.push_back(c);

                           return (found_col - this->column_names.begin() + this->index_names.size());
                       });

        table_view_out = table_view_out.select(col_indices);
    }

    // If the start/stop is different, then perform the slice
    if (start != 0 || stop != this->table_view.num_rows())
    {
        table_view_out = cudf::slice(table_view_out, {start, stop})[0];
    }

    // Create a new TableInfoData
    return {table_view_out, this->index_names, column_names};
}

}  // namespace morpheus
