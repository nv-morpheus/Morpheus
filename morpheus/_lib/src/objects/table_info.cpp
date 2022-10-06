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

#include "morpheus/objects/table_info.hpp"

#include "morpheus/utilities/type_util_detail.hpp"

#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/gil.h>       // for gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/stl.h>       // IWYU pragma: keep

#include <algorithm>  // for find, transform
#include <array>      // needed for pybind11::make_tuple
#include <cstddef>    // for size_t
#include <iterator>   // for back_insert_iterator, back_inserter
#include <memory>
#include <stdexcept>
#include <utility>
// IWYU pragma: no_include <pybind11/cast.h>

namespace morpheus {
/****** Component public implementations *******************/
/****** TableInfo****************************************/
TableInfo::TableInfo(std::shared_ptr<const IDataTable> parent,
                     cudf::table_view view,
                     std::vector<std::string> index_names,
                     std::vector<std::string> column_names) :
  m_parent(std::move(parent)),
  m_table_view(std::move(view)),
  m_index_names(std::move(index_names)),
  m_column_names(std::move(column_names))
{}

const pybind11::object &TableInfo::get_parent_table() const
{
    return m_parent->get_py_object();
}

const cudf::table_view &TableInfo::get_view() const
{
    return m_table_view;
}

std::vector<std::string> TableInfo::get_index_names() const
{
    return m_index_names;
}

std::vector<std::string> TableInfo::get_column_names() const
{
    return m_column_names;
}

cudf::size_type TableInfo::num_indices() const
{
    return this->get_index_names().size();
}

cudf::size_type TableInfo::num_columns() const
{
    return this->get_column_names().size();
}

cudf::size_type TableInfo::num_rows() const
{
    return this->m_table_view.num_rows();
}

pybind11::object TableInfo::as_py_object() const
{
    const auto offset = m_table_view.column(0).offset();
    const auto stop   = offset + this->num_rows();

    {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;

        auto df          = this->get_parent_table();
        auto index_slice = py::slice(py::int_(offset), py::int_(stop), py::none());
        return df.attr("loc")[py::make_tuple(df.attr("index")[index_slice], m_column_names)];
    }
}

void TableInfo::insert_columns(const std::vector<std::string> &column_names, const std::vector<TypeId> &column_types)
{
    CHECK(column_names.size() == column_types.size());
    const auto num_existing_cols = m_column_names.size();
    const auto num_rows          = m_table_view.num_rows();

    // TODO figure out how to do this without the gil
    {
        namespace py = pybind11;
        pybind11::gil_scoped_acquire gil;
        pybind11::object cupy_zeros = pybind11::module_::import("cupy").attr("zeros");

        auto table = get_parent_table();

        for (std::size_t i = 0; i < column_names.size(); ++i)
        {
            auto empty_array = cupy_zeros(num_rows, DataType(column_types[i]).type_str());
            table.attr("insert")(num_existing_cols + i, column_names[i], empty_array);
            m_column_names.push_back(column_names[i]);
        }
    }
}

void TableInfo::insert_missing_columns(const std::vector<std::string> &column_names,
                                       const std::vector<TypeId> &column_types)
{
    CHECK(column_names.size() == column_types.size());

    std::vector<std::string> missing_names;
    std::vector<TypeId> missing_types;
    for (std::size_t i = 0; i < column_names.size(); ++i)
    {
        if (std::find(m_column_names.begin(), m_column_names.end(), column_names[i]) == m_column_names.end())
        {
            missing_names.push_back(column_names[i]);
            missing_types.push_back(column_types[i]);
        }
    }

    if (!missing_names.empty())
    {
        insert_columns(missing_names, missing_types);
    }
}

const cudf::column_view &TableInfo::get_column(cudf::size_type idx) const
{
    if (idx < 0 || idx >= this->m_table_view.num_columns())
    {
        throw std::invalid_argument("idx must satisfy 0 <= idx < num_columns()");
    }

    return this->m_table_view.column(this->m_index_names.size() + idx);
}

TableInfo TableInfo::get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names) const
{
    std::vector<cudf::size_type> col_indices;

    std::vector<std::string> new_column_names;

    // Append the indices column idx by default
    for (cudf::size_type i = 0; i < this->m_index_names.size(); ++i)
    {
        col_indices.push_back(i);
    }

    std::transform(column_names.begin(),
                   column_names.end(),
                   std::back_inserter(col_indices),
                   [this, &new_column_names](const std::string &c) {
                       auto found_col = std::find(this->m_column_names.begin(), this->m_column_names.end(), c);

                       if (found_col == this->m_column_names.end())
                       {
                           throw std::runtime_error("Unknown column: " + c);
                       }

                       // Add the found column to the metadata
                       new_column_names.push_back(c);

                       return (found_col - this->m_column_names.begin() + this->num_indices());
                   });

    auto slice_rows = cudf::slice(m_table_view, {start, stop})[0];

    auto slice_cols = slice_rows.select(col_indices);

    return TableInfo(m_parent, slice_cols, m_index_names, new_column_names);
}
}  // namespace morpheus
