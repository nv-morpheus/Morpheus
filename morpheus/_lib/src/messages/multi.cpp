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

#include <morpheus/messages/multi.hpp>

#include <morpheus/messages/meta.hpp>
#include <morpheus/objects/table_info.hpp>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiMessage****************************************/
MultiMessage::MultiMessage(std::shared_ptr<morpheus::MessageMeta> m, size_t o, size_t c) :
  meta(std::move(m)),
  mess_offset(o),
  mess_count(c)
{}

TableInfo MultiMessage::get_meta()
{
    auto table_info = this->get_meta(std::vector<std::string>{});

    return table_info;
}

TableInfo MultiMessage::get_meta(const std::string &col_name)
{
    auto table_view = this->get_meta(std::vector<std::string>{col_name});

    return table_view;
}

TableInfo MultiMessage::get_meta(const std::vector<std::string> &column_names)
{
    TableInfo info = this->meta->get_info();

    TableInfo sliced_info = info.get_slice(this->mess_offset,
                                           this->mess_offset + this->mess_count,
                                           column_names.empty() ? info.get_column_names() : column_names);

    return sliced_info;
}

std::shared_ptr<MultiMessage> MultiMessage::get_slice(size_t start, size_t stop) const
{
    // This can only cast down
    return std::static_pointer_cast<MultiMessage>(this->internal_get_slice(start, stop));
}

std::shared_ptr<MultiMessage> MultiMessage::internal_get_slice(size_t start, size_t stop) const
{
    auto mess_start = this->mess_offset + start;
    auto mess_stop  = this->mess_offset + stop;

    return std::make_shared<MultiMessage>(this->meta, mess_start, mess_stop - mess_start);
}

std::shared_ptr<MultiMessage> MultiMessage::copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                        size_t num_selected_rows) const
{
    return std::static_pointer_cast<MultiMessage>(this->internal_copy_ranges(ranges, num_selected_rows));
}

std::shared_ptr<MultiMessage> MultiMessage::internal_copy_ranges(const std::vector<std::pair<size_t, size_t>> &ranges,
                                                                 size_t num_selected_rows) const
{
    auto msg_meta = copy_meta_ranges(ranges);
    return std::make_shared<MultiMessage>(msg_meta, 0, num_selected_rows);
}

std::shared_ptr<MessageMeta> MultiMessage::copy_meta_ranges(const std::vector<std::pair<size_t, size_t>> &ranges) const
{
    // copy ranges into a sequntial list of values
    // https://github.com/rapidsai/cudf/issues/11223
    std::vector<cudf::size_type> cudf_ranges;
    for (const auto &p : ranges)
    {
        cudf_ranges.push_back(static_cast<cudf::size_type>(p.first));
        cudf_ranges.push_back(static_cast<cudf::size_type>(p.second));
    }

    auto table_info                       = this->meta->get_info();
    std::vector<std::string> column_names = table_info.get_column_names();
    column_names.insert(column_names.begin(), std::string());  // cudf id col
    cudf::io::table_metadata metadata{std::move(column_names)};

    auto table_view                     = table_info.get_view();
    auto sliced_views                   = cudf::slice(table_view, cudf_ranges);
    cudf::io::table_with_metadata table = {cudf::concatenate(sliced_views, rmm::mr::get_current_device_resource()),
                                           std::move(metadata)};

    return MessageMeta::create_from_cpp(std::move(table), 1);
}

void MultiMessage::set_meta(const std::string &col_name, TensorObject tensor)
{
    set_meta(std::vector<std::string>{col_name}, std::vector<TensorObject>{tensor});
}

void MultiMessage::set_meta(const std::vector<std::string> &column_names, const std::vector<TensorObject> &tensors)
{
    std::vector<TypeId> tensor_types{tensors.size()};
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        tensor_types[i] = tensors[i].dtype().type_id();
    }

    TableInfo info = this->meta->get_info();
    info.insert_missing_columns(column_names, tensor_types);

    TableInfo table_meta = this->get_meta(column_names);
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        const auto cv          = table_meta.get_column(i);
        const auto table_type  = cv.type().id();
        const auto tensor_type = DType(tensor_types[i]).cudf_type_id();
        const auto row_stride  = tensors[i].stride(0);

        CHECK(tensors[i].count() == cv.size() && (table_type == tensor_type || (table_type == cudf::type_id::BOOL8 &&
                                                                                tensor_type == cudf::type_id::UINT8)));

        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            SRF_CHECK_CUDA(cudaMemcpy(const_cast<uint8_t *>(cv.data<uint8_t>()),
                                      tensors[i].data(),
                                      tensors[i].bytes(),
                                      cudaMemcpyDeviceToDevice));
        }
        else
        {
            const auto item_size = tensors[i].dtype().item_size();
            SRF_CHECK_CUDA(cudaMemcpy2D(const_cast<uint8_t *>(cv.data<uint8_t>()),
                                        item_size,
                                        tensors[i].data(),
                                        row_stride * item_size,
                                        item_size,
                                        cv.size(),
                                        cudaMemcpyDeviceToDevice));
        }
    }
}

std::vector<std::pair<TensorIndex, TensorIndex>> MultiMessage::apply_offset_to_ranges(
    std::size_t offset, const std::vector<std::pair<size_t, size_t>> &ranges) const
{
    std::vector<std::pair<TensorIndex, TensorIndex>> offset_ranges(ranges.size());
    std::transform(
        ranges.cbegin(), ranges.cend(), offset_ranges.begin(), [offset](const std::pair<size_t, size_t> range) {
            return std::pair{offset + range.first, offset + range.second};
        });

    return offset_ranges;
}

/****** MultiMessageInterfaceProxy *************************/
std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                               cudf::size_type mess_offset,
                                                               cudf::size_type mess_count)
{
    return std::make_shared<MultiMessage>(std::move(meta), mess_offset, mess_count);
}

std::shared_ptr<morpheus::MessageMeta> MultiMessageInterfaceProxy::meta(const MultiMessage &self)
{
    return self.meta;
}

std::size_t MultiMessageInterfaceProxy::mess_offset(const MultiMessage &self)
{
    return self.mess_offset;
}

std::size_t MultiMessageInterfaceProxy::mess_count(const MultiMessage &self)
{
    return self.mess_count;
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage &self)
{
    // Mimic this python code
    // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
    // value
    auto df = self.meta->get_py_table();

    auto index_slice = pybind11::slice(
        pybind11::int_(self.mess_offset), pybind11::int_(self.mess_offset + self.mess_count), pybind11::none());

    // Must do implicit conversion to pybind11::object here!!!
    pybind11::object df_slice = df.attr("loc")[df.attr("index")[index_slice]];

    return df_slice;
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage &self, std::string col_name)
{
    // Get the column and convert to cudf
    auto info = self.get_meta(col_name);

    return info.as_py_object();
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage &self, std::vector<std::string> columns)
{
    // Get the column and convert to cudf
    auto info = self.get_meta(columns);

    return info.as_py_object();
}

pybind11::object MultiMessageInterfaceProxy::get_meta_by_col(MultiMessage &self, pybind11::object columns)
{
    // // Get the column and convert to cudf
    // auto info = self.get_meta(columns);

    // auto py_table_struct = make_table_from_table_info(info, (PyObject*)info.get_parent_table().ptr());

    // if (!py_table_struct)
    // {
    //     throw pybind11::error_already_set();
    // }

    // pybind11::object py_table = pybind11::reinterpret_steal<pybind11::object>((PyObject*)py_table_struct);

    // return py_table;

    // Mimic this python code
    // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
    // value
    auto df = self.meta->get_py_table();

    auto index_slice = pybind11::slice(
        pybind11::int_(self.mess_offset), pybind11::int_(self.mess_offset + self.mess_count), pybind11::none());

    // Must do implicit conversion to pybind11::object here!!!
    pybind11::object df_slice = df.attr("loc")[pybind11::make_tuple(df.attr("index")[index_slice], columns)];

    return df_slice;
}

pybind11::object MultiMessageInterfaceProxy::get_meta_list(MultiMessage &self, pybind11::object col_name)
{
    std::vector<std::string> column_names;
    if (!col_name.is_none())
    {
        column_names.emplace_back(col_name.cast<std::string>());
    }

    auto info = self.get_meta(column_names);
    auto meta = info.as_py_object();
    if (!col_name.is_none())
    {  // needed to slice off the id column
        meta = meta[col_name];
    }

    auto arrow_tbl           = meta.attr("to_arrow")();
    pybind11::object py_list = arrow_tbl.attr("to_pylist")();
    return py_list;
}

void MultiMessageInterfaceProxy::set_meta(MultiMessage &self, pybind11::object columns, pybind11::object value)
{
    // Mimic this python code
    // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
    // value
    auto df = self.meta->get_py_table();

    auto index_slice = pybind11::slice(
        pybind11::int_(self.mess_offset), pybind11::int_(self.mess_offset + self.mess_count), pybind11::none());

    df.attr("loc")[pybind11::make_tuple(df.attr("index")[index_slice], columns)] = value;
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::get_slice(MultiMessage &self,
                                                                    std::size_t start,
                                                                    std::size_t stop)
{
    // Returns shared_ptr
    return self.get_slice(start, stop);
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::copy_ranges(
    MultiMessage &self, const std::vector<std::pair<size_t, size_t>> &ranges, pybind11::object num_selected_rows)
{
    std::size_t num_rows = 0;
    if (num_selected_rows.is_none())
    {
        for (const auto &range : ranges)
        {
            num_rows += range.second - range.first;
        }
    }
    else
    {
        num_rows = num_selected_rows.cast<std::size_t>();
    }
    return self.copy_ranges(ranges, num_rows);
}
}  // namespace morpheus
