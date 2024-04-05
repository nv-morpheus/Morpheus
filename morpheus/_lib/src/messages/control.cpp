/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/control.hpp"

#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/dtype.hpp"
#include "morpheus/objects/tensor_object.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cuda_runtime.h>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <mrc/cuda/common.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utils.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <regex>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace py = pybind11;
using namespace py::literals;

namespace morpheus {

const std::string ControlMessage::s_config_schema = R"()";

std::map<std::string, ControlMessageType> ControlMessage::s_task_type_map{{"inference", ControlMessageType::INFERENCE},
                                                                          {"training", ControlMessageType::TRAINING}};

ControlMessage::ControlMessage() : m_config({{"metadata", nlohmann::json::object()}}), m_tasks({}) {}

ControlMessage::ControlMessage(const nlohmann::json& _config) :
  m_config({{"metadata", nlohmann::json::object()}}),
  m_tasks({})
{
    config(_config);
}

ControlMessage::ControlMessage(const ControlMessage& other)
{
    m_config = other.m_config;
    m_tasks  = other.m_tasks;
}

const nlohmann::json& ControlMessage::config() const
{
    return m_config;
}

void ControlMessage::add_task(const std::string& task_type, const nlohmann::json& task)
{
    VLOG(20) << "Adding task of type " << task_type << " to control message" << task.dump(4);
    auto _task_type = s_task_type_map.contains(task_type) ? s_task_type_map[task_type] : ControlMessageType::NONE;

    if (this->task_type() == ControlMessageType::NONE)
    {
        this->task_type(_task_type);
    }

    if (_task_type != ControlMessageType::NONE and this->task_type() != _task_type)
    {
        throw std::runtime_error("Cannot add inference and training tasks to the same control message");
    }

    m_tasks[task_type].push_back(task);
}

bool ControlMessage::has_task(const std::string& task_type) const
{
    return m_tasks.contains(task_type) && m_tasks.at(task_type).size() > 0;
}

const nlohmann::json& ControlMessage::get_tasks() const
{
    return m_tasks;
}

std::vector<std::string> ControlMessage::list_metadata() const
{
    std::vector<std::string> key_list{};

    for (auto it = m_config["metadata"].begin(); it != m_config["metadata"].end(); ++it)
    {
        key_list.push_back(it.key());
    }

    return key_list;
}

void ControlMessage::set_metadata(const std::string& key, const nlohmann::json& value)
{
    if (m_config["metadata"].contains(key))
    {
        VLOG(20) << "Overwriting metadata key " << key << " with value " << value;
    }

    m_config["metadata"][key] = value;
}

bool ControlMessage::has_metadata(const std::string& key) const
{
    return m_config["metadata"].contains(key);
}

nlohmann::json ControlMessage::get_metadata() const
{
    auto metadata = m_config["metadata"];

    return metadata;
}

nlohmann::json ControlMessage::get_metadata(const std::string& key, bool fail_on_nonexist) const
{
    // Assuming m_metadata is a std::map<std::string, nlohmann::json> storing metadata
    auto metadata = m_config["metadata"];
    auto it       = metadata.find(key);
    if (it != metadata.end())
    {
        return metadata.at(key);
    }
    else if (fail_on_nonexist)
    {
        throw std::runtime_error("Metadata key does not exist: " + key);
    }

    return {};
}

nlohmann::json ControlMessage::remove_task(const std::string& task_type)
{
    auto& task_set = m_tasks.at(task_type);
    auto iter_task = task_set.begin();

    if (iter_task != task_set.end())
    {
        auto task = *iter_task;
        task_set.erase(iter_task);

        return task;
    }

    throw std::runtime_error("No tasks of type " + task_type + " found");
}

void ControlMessage::set_timestamp(const std::string& key, time_point_t timestamp_ns)
{
    // Insert or update the timestamp in the map
    m_timestamps[key] = timestamp_ns;
}

std::map<std::string, time_point_t> ControlMessage::filter_timestamp(const std::string& regex_filter)
{
    std::map<std::string, time_point_t> matching_timestamps;
    std::regex filter(regex_filter);

    for (const auto& [key, timestamp] : m_timestamps)
    {
        // Check if the key matches the regex
        if (std::regex_search(key, filter))
        {
            matching_timestamps[key] = timestamp;
        }
    }

    return matching_timestamps;
}

std::optional<time_point_t> ControlMessage::get_timestamp(const std::string& key, bool fail_if_nonexist)
{
    auto it = m_timestamps.find(key);
    if (it != m_timestamps.end())
    {
        return it->second;  // Return the found timestamp
    }
    else if (fail_if_nonexist)
    {
        throw std::runtime_error("Timestamp for the specified key does not exist.");
    }
    return std::nullopt;
}

void ControlMessage::config(const nlohmann::json& config)
{
    if (config.contains("type"))
    {
        auto task_type = config.at("type");
        auto _task_type =
            s_task_type_map.contains(task_type) ? s_task_type_map.at(task_type) : ControlMessageType::NONE;

        if (this->task_type() == ControlMessageType::NONE)
        {
            this->task_type(_task_type);
        }
    }

    if (config.contains("tasks"))
    {
        auto& tasks = config["tasks"];
        for (const auto& task : tasks)
        {
            add_task(task.at("type"), task.at("properties"));
        }
    }

    if (config.contains("metadata"))
    {
        auto& metadata = config["metadata"];
        for (auto it = metadata.begin(); it != metadata.end(); ++it)
        {
            set_metadata(it.key(), it.value());
        }
    }
}

std::shared_ptr<MessageMeta> ControlMessage::payload()
{
    return m_payload;
}

void ControlMessage::payload(const std::shared_ptr<MessageMeta>& payload)
{
    m_payload = payload;
}

std::shared_ptr<TensorMemory> ControlMessage::tensors()
{
    return m_tensors;
}

void ControlMessage::tensors(const std::shared_ptr<TensorMemory>& tensors)
{
    m_tensors = tensors;
}

void ControlMessage::set_meta(const std::string& col_name, TensorObject tensor)
{
    set_meta(std::vector<std::string>{col_name}, std::vector<TensorObject>{tensor});
}

TableInfo ControlMessage::get_meta()
{
    auto table_info = this->get_meta(std::vector<std::string>{});

    return table_info;
}

TableInfo ControlMessage::get_meta(const std::string& col_name)
{
    auto table_view = this->get_meta(std::vector<std::string>{col_name});

    return table_view;
}

TableInfo ControlMessage::get_meta(const std::vector<std::string>& column_names)
{
    TableInfo info = this->payload()->get_info();

    TableInfo sliced_info =
        info.get_slice(0, info.num_rows(), column_names.empty() ? info.get_column_names() : column_names);

    return sliced_info;
}

void ControlMessage::set_meta(const std::vector<std::string>& column_names, const std::vector<TensorObject>& tensors)
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

    for (std::size_t i = 0; i < tensors.size(); ++i)
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

ControlMessageType ControlMessage::task_type()
{
    return m_cm_type;
}

void ControlMessage::task_type(ControlMessageType type)
{
    m_cm_type = type;
}

/*** Proxy Implementations ***/

std::shared_ptr<ControlMessage> ControlMessageProxy::create(py::dict& config)
{
    return std::make_shared<ControlMessage>(mrc::pymrc::cast_from_pyobject(config));
}

std::shared_ptr<ControlMessage> ControlMessageProxy::create(std::shared_ptr<ControlMessage> other)
{
    return std::make_shared<ControlMessage>(*other);
}

std::shared_ptr<ControlMessage> ControlMessageProxy::copy(ControlMessage& self)
{
    return std::make_shared<ControlMessage>(self);
}

void ControlMessageProxy::add_task(ControlMessage& self, const std::string& task_type, py::dict& task)
{
    self.add_task(task_type, mrc::pymrc::cast_from_pyobject(task));
}

py::dict ControlMessageProxy::remove_task(ControlMessage& self, const std::string& task_type)
{
    auto task = self.remove_task(task_type);

    return mrc::pymrc::cast_from_json(task);
}

py::dict ControlMessageProxy::get_tasks(ControlMessage& self)
{
    return mrc::pymrc::cast_from_json(self.get_tasks());
}

py::dict ControlMessageProxy::config(ControlMessage& self)
{
    auto dict = mrc::pymrc::cast_from_json(self.config());

    return dict;
}

py::object ControlMessageProxy::get_metadata(ControlMessage& self,
                                             const py::object& key,
                                             pybind11::object default_value)
{
    if (key.is_none())
    {
        auto metadata = self.get_metadata();
        return mrc::pymrc::cast_from_json(metadata);
    }

    auto value = self.get_metadata(py::cast<std::string>(key), false);
    if (value.empty())
    {
        return default_value;
    }

    return mrc::pymrc::cast_from_json(value);
}

void ControlMessageProxy::set_metadata(ControlMessage& self, const std::string& key, pybind11::object& value)
{
    self.set_metadata(key, mrc::pymrc::cast_from_pyobject(value));
}

py::list ControlMessageProxy::list_metadata(ControlMessage& self)
{
    auto keys = self.list_metadata();
    py::list py_keys;
    for (const auto& key : keys)
    {
        py_keys.append(py::str(key));
    }
    return py_keys;
}

py::dict ControlMessageProxy::filter_timestamp(ControlMessage& self, const std::string& regex_filter)
{
    auto cpp_map = self.filter_timestamp(regex_filter);
    py::dict py_dict;
    for (const auto& [key, timestamp] : cpp_map)
    {
        // Directly use the timestamp as datetime.datetime in Python
        py_dict[py::str(key)] = timestamp;
    }
    return py_dict;
}

// Get a specific timestamp and return it as datetime.datetime or None
py::object ControlMessageProxy::get_timestamp(ControlMessage& self, const std::string& key, bool fail_if_nonexist)
{
    try
    {
        auto timestamp_opt = self.get_timestamp(key, fail_if_nonexist);
        if (timestamp_opt)
        {
            // Directly return the timestamp as datetime.datetime in Python
            return py::cast(*timestamp_opt);
        }

        return py::none();
    } catch (const std::runtime_error& e)
    {
        if (fail_if_nonexist)
        {
            throw py::value_error(e.what());
        }
        return py::none();
    }
}

// Set a timestamp using a datetime.datetime object from Python
void ControlMessageProxy::set_timestamp(ControlMessage& self, const std::string& key, py::object timestamp_ns)
{
    if (!py::isinstance<py::none>(timestamp_ns))
    {
        // Convert Python datetime.datetime to std::chrono::system_clock::time_point before setting
        auto _timestamp_ns = timestamp_ns.cast<time_point_t>();
        self.set_timestamp(key, _timestamp_ns);
    }
    else
    {
        throw std::runtime_error("Timestamp cannot be None");
    }
}

void ControlMessageProxy::config(ControlMessage& self, py::dict& config)
{
    self.config(mrc::pymrc::cast_from_pyobject(config));
}

void ControlMessageProxy::payload_from_python_meta(ControlMessage& self, const pybind11::object& meta)
{
    self.payload(MessageMetaInterfaceProxy::init_python_meta(meta));
}

pybind11::object ControlMessageProxy::get_meta(ControlMessage& self)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta();

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object ControlMessageProxy::get_meta(ControlMessage& self, std::string col_name)
{
    TableInfo info;

    {
        // Need to release the GIL before calling `get_meta()`
        pybind11::gil_scoped_release no_gil;

        // Get the column and convert to cudf
        info = self.get_meta();
    }

    auto py_table = CudfHelper::table_from_table_info(info);

    // Now convert it to a series by selecting only the column
    return py_table[col_name.c_str()];
}

pybind11::object ControlMessageProxy::get_meta(ControlMessage& self, std::vector<std::string> columns)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    // Get the column and convert to cudf
    auto info = self.get_meta(columns);

    // Convert to a python datatable. Automatically gets the GIL
    return CudfHelper::table_from_table_info(info);
}

pybind11::object ControlMessageProxy::get_meta(ControlMessage& self, pybind11::none none_obj)
{
    // Just offload to the overload without columns. This overload is needed to match the python interface
    return ControlMessageProxy::get_meta(self);
}

std::tuple<py::object, py::object> get_indexers(ControlMessage& self,
                                                py::object df,
                                                py::object columns,
                                                cudf::size_type num_rows)
{
    auto row_indexer = pybind11::slice(pybind11::int_(0), pybind11::int_(num_rows), pybind11::none());

    if (columns.is_none())
    {
        columns = df.attr("columns").attr("to_list")();
    }
    else if (pybind11::isinstance<pybind11::str>(columns))
    {
        // Convert a single string into a list so all versions return tables, not series
        pybind11::list col_list;

        col_list.append(columns);

        columns = std::move(col_list);
    }

    auto column_indexer = df.attr("columns").attr("get_indexer_for")(columns);

    return std::make_tuple(row_indexer, column_indexer);
}

void ControlMessageProxy::set_meta(ControlMessage& self, pybind11::object columns, pybind11::object value)
{
    // Need to release the GIL before calling `get_meta()`
    pybind11::gil_scoped_release no_gil;

    auto mutable_info = self.payload()->get_mutable_info();
    auto num_rows     = mutable_info.num_rows();

    // Need the GIL for the remainder
    pybind11::gil_scoped_acquire gil;

    auto pdf = mutable_info.checkout_obj();
    auto& df = *pdf;

    auto [row_indexer, column_indexer] = get_indexers(self, df, columns, num_rows);

    // Check to see if this is adding a column. If so, we need to use .loc instead of .iloc
    if (column_indexer.contains(-1))
    {
        // cudf is really bad at adding new columns. Need to use loc with a unique and monotonic index
        py::object saved_index = df.attr("index");

        // Check to see if we can use slices
        if (!(saved_index.attr("is_unique").cast<bool>() && (saved_index.attr("is_monotonic_increasing").cast<bool>() ||
                                                             saved_index.attr("is_monotonic_decreasing").cast<bool>())))
        {
            df.attr("reset_index")("drop"_a = true, "inplace"_a = true);
        }
        else
        {
            // Erase the saved index so we dont reset it
            saved_index = py::none();
        }

        // Perform the update via slices
        df.attr("loc")[pybind11::make_tuple(df.attr("index")[row_indexer], columns)] = value;

        // Reset the index if we changed it
        if (!saved_index.is_none())
        {
            df.attr("set_index")(saved_index, "inplace"_a = true);
        }
    }
    else
    {
        // If we only have one column, convert it to a series (broadcasts work with more types on a series)
        if (pybind11::len(column_indexer) == 1)
        {
            column_indexer = column_indexer.cast<py::list>()[0];
        }

        try
        {
            // Use iloc
            df.attr("iloc")[pybind11::make_tuple(row_indexer, column_indexer)] = value;
        } catch (py::error_already_set)
        {
            // Try this as a fallback. Works better for strings. See issue #286
            df[columns].attr("iloc")[row_indexer] = value;
        }
    }

    mutable_info.return_obj(std::move(pdf));
}

}  // namespace morpheus
