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

#include "morpheus/messages/multi.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/dtype.hpp"  // for TypeId, DType
#include "morpheus/objects/table_info.hpp"
#include "morpheus/objects/tensor_object.hpp"

#include <cuda_runtime.h>               // for cudaMemcpy, cudaMemcpy2D, cudaMemcpyDeviceToDevice
#include <cudf/column/column_view.hpp>  // for column_view
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/mr/device/per_device_resource.hpp>  // for get_current_device_resource

#include <algorithm>  // for transform
#include <array>      // needed for pybind11::make_tuple
#include <cstdint>    // for uint8_t
#include <sstream>
#include <stdexcept>  // for runtime_error
// IWYU pragma: no_include <unordered_map>

namespace morpheus {
/****** Component public implementations *******************/
/****** MultiMessage****************************************/
MultiMessage::MultiMessage(std::shared_ptr<morpheus::MessageMeta> m, TensorIndex o, TensorIndex c) :
  meta(std::move(m)),
  mess_offset(o),
  mess_count(c)
{}

TableInfo MultiMessage::get_meta()
{
    auto table_info = this->get_meta(std::vector<std::string>{});

    return table_info;
}

TableInfo MultiMessage::get_meta(const std::string& col_name)
{
    auto table_view = this->get_meta(std::vector<std::string>{col_name});

    return table_view;
}

TableInfo MultiMessage::get_meta(const std::vector<std::string>& column_names)
{
    TableInfo info = this->meta->get_info();

    TableInfo sliced_info = info.get_slice(this->mess_offset,
                                           this->mess_offset + this->mess_count,
                                           column_names.empty() ? info.get_column_names() : column_names);

    return sliced_info;
}

void MultiMessage::get_slice_impl(std::shared_ptr<MultiMessage> new_message, TensorIndex start, TensorIndex stop) const
{
    new_message->mess_offset = this->mess_offset + start;
    new_message->mess_count  = this->mess_offset + stop - new_message->mess_offset;
}

void MultiMessage::copy_ranges_impl(std::shared_ptr<MultiMessage> new_message,
                                    const std::vector<RangeType>& ranges,
                                    TensorIndex num_selected_rows) const
{
    new_message->mess_offset = 0;
    new_message->mess_count  = num_selected_rows;
    new_message->meta        = copy_meta_ranges(ranges);
}

std::shared_ptr<MessageMeta> MultiMessage::copy_meta_ranges(const std::vector<RangeType>& ranges) const
{
    // copy ranges into a sequntial list of values
    // https://github.com/rapidsai/cudf/issues/11223
    std::vector<TensorIndex> cudf_ranges;
    for (const auto& p : ranges)
    {
        // Append the message offset to the range here
        cudf_ranges.push_back(p.first + this->mess_offset);
        cudf_ranges.push_back(p.second + this->mess_offset);
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

void MultiMessage::set_meta(const std::string& col_name, TensorObject tensor)
{
    set_meta(std::vector<std::string>{col_name}, std::vector<TensorObject>{tensor});
}

void MultiMessage::set_meta(const std::vector<std::string>& column_names, const std::vector<TensorObject>& tensors)
{
    TableInfo table_meta;
    try
    {
        table_meta = this->get_meta(column_names);
    } catch (const std::runtime_error& e)
    {
        std::ostringstream err_msg;
        err_msg << e.what() << " Ensure that the stage that needs this column has populated the '_needed_columns' "
                << "attribute and that at least one stage in the current segment is using the PreallocatorMixin to "
                << "ensure all needed columns have been allocated.";
        throw std::runtime_error(err_msg.str());
    }

    for (TensorIndex i = 0; i < tensors.size(); ++i)
    {
        const auto& cv            = table_meta.get_column(i);
        const auto table_type_id  = cv.type().id();
        const auto tensor_type    = DType(tensors[i].dtype());
        const auto tensor_type_id = tensor_type.cudf_type_id();
        const auto row_stride     = tensors[i].stride(0);

        CHECK(tensors[i].count() == cv.size() &&
              (table_type_id == tensor_type_id ||
               (table_type_id == cudf::type_id::BOOL8 && tensor_type_id == cudf::type_id::UINT8)));

        const auto item_size = tensors[i].dtype().item_size();

        // Dont use cv.data<>() here since that does not account for the size of each element
        auto data_start = const_cast<uint8_t*>(cv.head<uint8_t>()) + cv.offset() * item_size;

        if (row_stride == 1)
        {
            // column major just use cudaMemcpy
            MRC_CHECK_CUDA(cudaMemcpy(data_start, tensors[i].data(), tensors[i].bytes(), cudaMemcpyDeviceToDevice));
        }
        else
        {
            MRC_CHECK_CUDA(cudaMemcpy2D(data_start,
                                        item_size,
                                        tensors[i].data(),
                                        row_stride * item_size,
                                        item_size,
                                        cv.size(),
                                        cudaMemcpyDeviceToDevice));
        }
    }
}

std::vector<RangeType> MultiMessage::apply_offset_to_ranges(TensorIndex offset,
                                                            const std::vector<RangeType>& ranges) const
{
    std::vector<RangeType> offset_ranges(ranges.size());
    std::transform(ranges.cbegin(),
                   ranges.cend(),
                   offset_ranges.begin(),
                   [offset](const std::pair<TensorIndex, TensorIndex> range) {
                       return std::pair{offset + range.first, offset + range.second};
                   });

    return offset_ranges;
}

/****** MultiMessageInterfaceProxy *************************/
std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::init(std::shared_ptr<MessageMeta> meta,
                                                               TensorIndex mess_offset,
                                                               TensorIndex mess_count)
{
    return std::make_shared<MultiMessage>(std::move(meta), mess_offset, mess_count);
}

std::shared_ptr<morpheus::MessageMeta> MultiMessageInterfaceProxy::meta(const MultiMessage& self)
{
    return self.meta;
}

TensorIndex MultiMessageInterfaceProxy::mess_offset(const MultiMessage& self)
{
    return self.mess_offset;
}

TensorIndex MultiMessageInterfaceProxy::mess_count(const MultiMessage& self)
{
    return self.mess_count;
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta();

    return info.copy_to_py_object();
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self, std::string col_name)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta(col_name);

    return info.copy_to_py_object();
}

pybind11::object MultiMessageInterfaceProxy::get_meta(MultiMessage& self, std::vector<std::string> columns)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta(columns);

    return info.copy_to_py_object();
}

pybind11::object MultiMessageInterfaceProxy::get_meta_list(MultiMessage& self, pybind11::object col_name)
{
    std::vector<std::string> column_names;
    if (!col_name.is_none())
    {
        column_names.emplace_back(col_name.cast<std::string>());
    }

    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto info = self.get_meta(column_names);

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto meta = info.copy_to_py_object();

    if (!col_name.is_none())
    {  // needed to slice off the id column
        meta = meta[col_name];
    }

    auto arrow_tbl           = meta.attr("to_arrow")();
    pybind11::object py_list = arrow_tbl.attr("to_pylist")();

    return py_list;
}

void MultiMessageInterfaceProxy::set_meta(MultiMessage& self, pybind11::object columns, pybind11::object value)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto mutable_info = self.meta->get_mutable_info();

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto df = mutable_info.checkout_obj();

    auto index_slice = pybind11::slice(
        pybind11::int_(self.mess_offset), pybind11::int_(self.mess_offset + self.mess_count), pybind11::none());

    // Mimic this python code
    // self.meta.df.loc[self.meta.df.index[self.mess_offset:self.mess_offset + self.mess_count], columns] =
    // value
    df.attr("loc")[pybind11::make_tuple(df.attr("index")[index_slice], columns)] = value;

    mutable_info.return_obj(std::move(df));
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::get_slice(MultiMessage& self,
                                                                    TensorIndex start,
                                                                    TensorIndex stop)
{
    // Need to drop the GIL before calling any methods on the C++ object
    pybind11::gil_scoped_release no_gil;

    // Returns shared_ptr
    return self.get_slice(start, stop);
}

std::shared_ptr<MultiMessage> MultiMessageInterfaceProxy::copy_ranges(MultiMessage& self,
                                                                      const std::vector<RangeType>& ranges,
                                                                      pybind11::object num_selected_rows)
{
    TensorIndex num_rows = 0;
    if (num_selected_rows.is_none())
    {
        for (const auto& range : ranges)
        {
            num_rows += range.second - range.first;
        }
    }
    else
    {
        num_rows = num_selected_rows.cast<TensorIndex>();
    }

    // Need to drop the GIL before calling any methods on the C++ object
    pybind11::gil_scoped_release no_gil;

    return self.copy_ranges(ranges, num_rows);
}
}  // namespace morpheus
