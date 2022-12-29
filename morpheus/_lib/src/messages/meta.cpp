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

#include "morpheus/messages/meta.hpp"

#include "morpheus/objects/python_data_table.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"
#include "morpheus/utilities/table_util.hpp"

#include <cudf/io/types.hpp>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <utility>

namespace morpheus {
/******* Component-private Classes *********************/
/******* MessageMetaImpl *******************************/
/*
struct MessageMetaImpl {
    virtual pybind11::object get_py_table() const = 0;

    virtual TableInfo get_info() const = 0;
};
*/

/******* MessageMetaPyImpl *****************************/
// struct MessageMetaPyImpl : public MessageMetaImpl
// {
//     MessageMetaPyImpl(pybind11::object&& pydf) : m_pydf(std::move(pydf)) {}

//     MessageMetaPyImpl(cudf::io::table_with_metadata&& table) : m_pydf(std::move(cpp_to_py(std::move(table)))) {}

//     pybind11::object get_py_table() const override
//     {
//         return m_pydf;
//     }

//     TableInfo get_info() const override
//     {
//         pybind11::gil_scoped_acquire gil;

//         return make_table_info_from_table((PyTable*)this->m_pydf.ptr());
//     }

//     pybind11::object m_pydf;
// };

// struct MessageMetaCppImpl : public MessageMetaImpl
// {
//     MessageMetaCppImpl(cudf::io::table_with_metadata&& table) : m_table(std::move(table)) {}

//     pybind11::object get_py_table() const override
//     {
//         pybind11::gil_scoped_acquire gil;

//         // Get a python object from this data table
//         pybind11::object py_datatable = pybind11::cast(m_data_table);

//         // Now convert to a python TableInfo object
//         auto converted_table = pybind11::reinterpret_steal<pybind11::object>(
//             (PyObject*)make_table_from_datatable(m_data_table, (PyObject*)py_datatable.ptr()));

//         return converted_table;
//     }
//     TableInfo get_info() const override
//     {
//         return TableInfo(m_data_table);
//     }

//     std::shared_ptr<DataTable> m_data_table;
// };

// std::unique_ptr<MessageMetaImpl> m_data;

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

std::shared_ptr<MessageMeta> MessageMeta::create_from_python(pybind11::object&& data_table)
{
    auto data = std::make_unique<PyDataTable>(std::move(data_table));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

std::shared_ptr<MessageMeta> MessageMeta::create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                          int index_col_count)
{
    // Convert to py first
    pybind11::object py_dt = cpp_to_py(std::move(data_table), index_col_count);

    auto data = std::make_unique<PyDataTable>(std::move(py_dt));

    return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
}

MessageMeta::MessageMeta(std::shared_ptr<IDataTable> data) : m_data(std::move(data)) {}

pybind11::object MessageMeta::cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count)
{
    pybind11::gil_scoped_acquire gil;

    // Now convert to a python TableInfo object
    auto converted_table = proxy_table_from_table_with_metadata(std::move(table), index_col_count);

    // VLOG(10) << "Table. Num Col: " << converted_table.attr("_num_columns").str().cast<std::string>()
    //          << ", Num Ind: " << converted_table.attr("_num_columns").cast<std::string>()
    //          << ", Rows: " << converted_table.attr("_num_rows").cast<std::string>();
    // pybind11::print("Table Created. Num Rows: {}, Num Cols: {}, Num Ind: {}",
    //           converted_table.attr("_num_rows"),
    //           converted_table.attr("_num_columns"),
    //           converted_table.attr("_num_indices"));

    return converted_table;
}

/********** MutableCtxMgr **********/
MutableCtxMgr::MutableCtxMgr(MutableTableInfo&& table) : m_table{std::move(table)} {};

pybind11::object MutableCtxMgr::enter()
{
    std::cout << "__enter__"
              << " - " << std::flush;
    m_py_table = m_table.checkout_obj();
    return m_py_table;
}

void MutableCtxMgr::exit(const pybind11::object& type, const pybind11::object& value, const pybind11::object& traceback)
{
    std::cout << " - "
              << "__exit__" << std::endl;
    m_table.return_obj(std::move(m_py_table));
}

/********** MessageMetaInterfaceProxy **********/
std::shared_ptr<MessageMeta> MessageMetaInterfaceProxy::init_python(pybind11::object&& data_frame)
{
    return MessageMeta::create_from_python(std::move(data_frame));
}

cudf::size_type MessageMetaInterfaceProxy::count(MessageMeta& self)
{
    return self.count();
}

pybind11::object MessageMetaInterfaceProxy::get_data_frame(MessageMeta& self)
{
    // Release any GIL
    pybind11::gil_scoped_release no_gil;

    // return py_table;
    return self.get_info().copy_to_py_object();
}

void MessageMetaInterfaceProxy::set_data_frame(MessageMeta& self, const pybind11::object& new_df)
{
    // Release any GIL
    pybind11::gil_scoped_release no_gil;

    auto mutable_info = self.get_mutable_info();

    LOG(FATAL) << "Not implemented yet";

    // mutable_info.py_obj() = new_df;

    // pybind11::object obj = mutable_info.py_obj();

    // // // Get the column and convert to cudf
    // // auto py_table_struct = make_table_from_view_and_meta(self.m_pydf;.tbl->view(),
    // // self.m_pydf;.metadata); py::object py_table  =
    // // py::reinterpret_steal<py::object>((PyObject*)py_table_struct);

    // // // py_col.inc_ref();

    // // return py_table;
    // return self.get_py_table();
}

MutableCtxMgr MessageMetaInterfaceProxy::mutable_dataframe(MessageMeta& self)
{
    // Release any GIL
    pybind11::gil_scoped_release no_gil;
    return {self.get_mutable_info()};
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
