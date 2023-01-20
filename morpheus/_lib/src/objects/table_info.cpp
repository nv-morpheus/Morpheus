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

#include "morpheus/objects/table_info.hpp"

#include "morpheus/objects/dtype.hpp"

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
#include <shared_mutex>
#include <stdexcept>
#include <utility>
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
                               throw std::runtime_error("Unknown column: " + c);
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

/****** Component public implementations *******************/
/****** TableInfoBase****************************************/
TableInfoBase::TableInfoBase(std::shared_ptr<const IDataTable> parent, TableInfoData data) :
  m_parent(std::move(parent)),
  m_data(std::move(data))
{}

const cudf::table_view& TableInfoBase::get_view() const
{
    return m_data.table_view;
}

std::vector<std::string> TableInfoBase::get_index_names() const
{
    return m_data.index_names;
}

std::vector<std::string> TableInfoBase::get_column_names() const
{
    return m_data.column_names;
}

cudf::size_type TableInfoBase::num_indices() const
{
    return this->get_index_names().size();
}

cudf::size_type TableInfoBase::num_columns() const
{
    return this->get_column_names().size();
}

cudf::size_type TableInfoBase::num_rows() const
{
    return m_data.table_view.num_rows();
}

pybind11::object TableInfoBase::copy_to_py_object() const
{
    const auto offset = m_data.table_view.column(0).offset();
    const auto stop   = offset + this->num_rows();

    {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;

        auto df = m_parent->get_py_object();

        // Compute the DF slice in python
        auto index_slice = py::slice(py::int_(offset), py::int_(stop), py::none());

        auto py_slice = df.attr("loc")[py::make_tuple(df.attr("index")[index_slice], m_data.column_names)];

        auto copied_df = py_slice.attr("copy")();

        return copied_df;
    }
}

const cudf::column_view& TableInfoBase::get_column(cudf::size_type idx) const
{
    if (idx < 0 || idx >= this->m_data.table_view.num_columns())
    {
        throw std::invalid_argument("idx must satisfy 0 <= idx < num_columns()");
    }

    return this->m_data.table_view.column(this->m_data.index_names.size() + idx);
}

const std::shared_ptr<const IDataTable>& TableInfoBase::get_parent() const
{
    return m_parent;
}

TableInfoData& TableInfoBase::get_data()
{
    return m_data;
}

const TableInfoData& TableInfoBase::get_data() const
{
    return m_data;
}

TableInfo::TableInfo(std::shared_ptr<const IDataTable> parent,
                     std::shared_lock<std::shared_mutex> lock,
                     TableInfoData data) :
  TableInfoBase(parent, std::move(data)),
  m_lock(std::move(lock))
{}

TableInfo TableInfo::get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names) const
{
    // Create a new Table info, (cloning the shared_lock)
    return {this->get_parent(),
            std::shared_lock<std::shared_mutex>(*m_lock.mutex()),
            this->get_data().get_slice(start, stop, column_names)};
}

MutableTableInfo::MutableTableInfo(std::shared_ptr<const IDataTable> parent,
                                   std::unique_lock<std::shared_mutex> lock,
                                   TableInfoData data) :
  TableInfoBase(parent, std::move(data)),
  m_lock(std::move(lock))
{}

MutableTableInfo::~MutableTableInfo()
{
    if (m_checked_out_ref_count >= 0)
    {
        LOG(FATAL) << "Checked out python object was not returned before MutableTableInfo went out of scope";
    }
}

void MutableTableInfo::insert_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    const auto num_existing_cols = this->get_data().column_names.size();
    const auto num_rows          = this->get_data().table_view.num_rows();

    // TODO figure out how to do this without the gil
    {
        namespace py = pybind11;
        pybind11::gil_scoped_acquire gil;
        pybind11::object cudf_scalar = pybind11::module_::import("cudf").attr("Scalar");

        auto table = this->get_parent()->get_py_object();

        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            auto scalar = cudf_scalar(0, std::get<1>(columns[i]).type_str());
            table.attr("insert")(num_existing_cols + i, std::get<0>(columns[i]), scalar);
            this->get_data().column_names.push_back(std::get<0>(columns[i]));
        }
    }
}

void MutableTableInfo::insert_missing_columns(const std::vector<std::tuple<std::string, morpheus::DType>>& columns)
{
    std::vector<std::tuple<std::string, morpheus::DType>> missing_columns;
    for (const auto& column : columns)
    {
        if (std::find(this->get_data().column_names.begin(),
                      this->get_data().column_names.end(),
                      std::get<0>(column)) == this->get_data().column_names.end())
        {
            missing_columns.push_back(column);
        }
    }

    if (!missing_columns.empty())
    {
        insert_columns(missing_columns);
    }
}

pybind11::object MutableTableInfo::checkout_obj()
{
    // Get a copy increasing the ref count
    pybind11::object checked_out_obj = this->get_parent()->get_py_object();

    m_checked_out_ref_count = checked_out_obj.ref_count();

    return checked_out_obj;
}

void MutableTableInfo::return_obj(pybind11::object&& obj)
{
    m_checked_out_ref_count = -1;
}

}  // namespace morpheus
