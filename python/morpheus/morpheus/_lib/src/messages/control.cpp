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

#include "morpheus/messages/meta.hpp"  // for MessageMeta, MessageMetaInterfaceProxy

#include <glog/logging.h>       // for COMPACT_GOOGLE_LOG_INFO, LogMessage, VLOG
#include <nlohmann/json.hpp>    // for basic_json, json_ref, iter_impl, operator<<
#include <pybind11/chrono.h>    // IWYU pragma: keep
#include <pybind11/pybind11.h>  // for cast, object::cast
#include <pybind11/pytypes.h>   // for object, none, dict, isinstance, list, str, value_error, generic_item
#include <pymrc/utils.hpp>      // for cast_from_pyobject

#include <optional>   // for optional, nullopt
#include <ostream>    // for basic_ostream, operator<<
#include <regex>      // for regex_search, regex
#include <stdexcept>  // for runtime_error
#include <utility>    // for pair

namespace py = pybind11;
using namespace py::literals;

namespace morpheus {

const std::string ControlMessage::s_config_schema = R"()";

std::map<std::string, ControlMessageType> ControlMessage::s_task_type_map{{"inference", ControlMessageType::INFERENCE},
                                                                          {"training", ControlMessageType::TRAINING}};

ControlMessage::ControlMessage() : m_config({{"metadata", morpheus::utilities::json_t::object()}}), m_tasks({}) {}

ControlMessage::ControlMessage(const morpheus::utilities::json_t& _config) :
  m_config({{"metadata", morpheus::utilities::json_t::object()}}),
  m_tasks({})
{
    config(_config);
}

ControlMessage::ControlMessage(const ControlMessage& other)
{
    m_config = other.m_config;
    m_tasks  = other.m_tasks;
}

const morpheus::utilities::json_t& ControlMessage::config() const
{
    return m_config;
}

void ControlMessage::add_task(const std::string& task_type, const morpheus::utilities::json_t& task)
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

const morpheus::utilities::json_t& ControlMessage::get_tasks() const
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

void ControlMessage::set_metadata(const std::string& key, const morpheus::utilities::json_t& value)
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

morpheus::utilities::json_t ControlMessage::get_metadata() const
{
    auto metadata = m_config["metadata"];

    return metadata;
}

morpheus::utilities::json_t ControlMessage::get_metadata(const std::string& key, bool fail_on_nonexist) const
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

morpheus::utilities::json_t ControlMessage::remove_task(const std::string& task_type)
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

void ControlMessage::config(const morpheus::utilities::json_t& config)
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

py::object ControlMessageProxy::get_metadata(ControlMessage& self,
                                             const py::object& key,
                                             pybind11::object default_value)
{
    if (key.is_none())
    {
        auto metadata = self.get_metadata();
        return cast_from_json(metadata);
    }

    auto value = self.get_metadata(py::cast<std::string>(key), false);
    if (value.empty())
    {
        return default_value;
    }

    return cast_from_json(value);
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

void ControlMessageProxy::payload_from_python_meta(ControlMessage& self, const pybind11::object& meta)
{
    self.payload(MessageMetaInterfaceProxy::init_python_meta(meta));
}

}  // namespace morpheus
