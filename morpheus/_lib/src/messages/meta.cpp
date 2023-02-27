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

#include "morpheus/messages/meta.hpp"

#include "morpheus/objects/mutable_table_ctx_mgr.hpp"
#include "morpheus/objects/python_data_table.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/table_util.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>
#include <warnings.h>

#include <memory>
#include <utility>

namespace morpheus {

namespace py = pybind11;

/****** Component public implementations *******************/
/****** MessageMeta ****************************************/

size_t MessageMeta::count() const
{
    return m_data->count();
}

TableInfo MessageMeta::get_info() const
{
    return this->m_data->get_info();
}

MutableTableInfo MessageMeta::get_mutable_info() const
{
    return this->m_data->get_mutable_info();
}

std::shared_ptr<MessageMeta> MessageMeta::create_from_python(py::object&& data_table)
{
    auto data = std::make_unique<PyDataTable>(std::move(data_table));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

std::shared_ptr<MessageMeta> MessageMeta::create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                          int index_col_count)
{
    // Convert to py first
    py::object py_dt = cpp_to_py(std::move(data_table), index_col_count);

    auto data = std::make_unique<PyDataTable>(std::move(py_dt));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

MessageMeta::MessageMeta(std::shared_ptr<IDataTable> data) : m_data(std::move(data)) {}

py::object MessageMeta::cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count)
{
    py::gil_scoped_acquire gil;

    // Now convert to a python TableInfo object
    auto converted_table = proxy_table_from_table_with_metadata(std::move(table), index_col_count);

    // VLOG(10) << "Table. Num Col: " << converted_table.attr("_num_columns").str().cast<std::string>()
    //          << ", Num Ind: " << converted_table.attr("_num_columns").cast<std::string>()
    //          << ", Rows: " << converted_table.attr("_num_rows").cast<std::string>();
    // py::print("Table Created. Num Rows: {}, Num Cols: {}, Num Ind: {}",
    //           converted_table.attr("_num_rows"),
    //           converted_table.attr("_num_columns"),
    //           converted_table.attr("_num_indices"));

    return converted_table;
}

/********** MessageMetaInterfaceProxy **********/
std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_python(py::object&& data_frame)
{
    return MessageMeta::create_from_python(std::move(data_frame));
}

cudf::size_type MessageMetaInterfaceProxy::count(MessageMeta& self)
{
    return self.count();
}

py::object MessageMetaInterfaceProxy::get_data_frame(MessageMeta& self)
{
    // Release any GIL
    py::gil_scoped_release no_gil;

    // return py_table;
    return self.get_info().copy_to_py_object();
}

py::object MessageMetaInterfaceProxy::df_property(MessageMeta& self)
{
    PyErr_WarnEx(
        PyExc_DeprecationWarning,
        "Warning the df property returns a copy, please use the copy_dataframe method or the mutable_dataframe "
        "context manager to modify the DataFrame in-place instead.",
        1);

    return MessageMetaInterfaceProxy::get_data_frame(self);
}

MutableTableCtxMgr MessageMetaInterfaceProxy::mutable_dataframe(MessageMeta& self)
{
    // Release any GIL
    py::gil_scoped_release no_gil;
    return {self};
}

std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_cpp(const std::string& filename)
{
    // Load the file
    auto df_with_meta = CuDFTableUtil::load_table(filename);

    return MessageMeta::create_from_cpp(std::move(df_with_meta));
}

SlicedMessageMeta::SlicedMessageMeta(std::shared_ptr<MessageMeta> other,
                                     cudf::size_type start,
                                     cudf::size_type stop,
                                     std::vector<std::string> columns) :
  MessageMeta(*other),
  m_start(start),
  m_stop(stop),
  m_column_names(std::move(columns))
{}

TableInfo SlicedMessageMeta::get_info() const
{
    return this->m_data->get_info(m_start, m_stop, m_column_names);
}

MutableTableInfo SlicedMessageMeta::get_mutable_info() const
{
    return this->m_data->get_mutable_info(m_start, m_stop, m_column_names);
}

}  // namespace morpheus
