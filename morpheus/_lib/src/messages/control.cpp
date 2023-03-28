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

#include "morpheus/messages/control.hpp"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>

namespace py = pybind11;

namespace morpheus {

const std::string ControlMessage::s_config_schema = R"()";

std::map<std::string, ControlMessageType> ControlMessage::s_task_type_map{{"inference", ControlMessageType::INFERENCE},
                                                                          {"training", ControlMessageType::TRAINING}};

ControlMessage::ControlMessage() : m_config({{"metadata", nlohmann::json::object()}}) {}

ControlMessage::ControlMessage(const nlohmann::json& _config) : m_config({{"metadata", nlohmann::json::object()}})
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
    // TODO(Devin) Schema check
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

const nlohmann::json ControlMessage::get_metadata(const std::string& key) const
{
    return m_config["metadata"].at(key);
}

const nlohmann::json ControlMessage::remove_task(const std::string& task_type)
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
    // auto temp = std::move(m_payload);
    //  TODO(Devin): Decide if we copy or steal the payload
    //  m_payload = nullptr;

    return m_payload;
}

void ControlMessage::payload(const std::shared_ptr<MessageMeta>& payload)
{
    m_payload = payload;
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

py::dict ControlMessageProxy::config(ControlMessage& self)
{
    auto dict = mrc::pymrc::cast_from_json(self.config());

    return dict;
}

py::object ControlMessageProxy::get_metadata(ControlMessage& self, const std::string& key)
{
    auto dict = mrc::pymrc::cast_from_json(self.get_metadata(key));

    return dict;
}

void ControlMessageProxy::set_metadata(ControlMessage& self, const std::string& key, pybind11::object& value)
{
    self.set_metadata(key, mrc::pymrc::cast_from_pyobject(value));
}

void ControlMessageProxy::config(ControlMessage& self, py::dict& config)
{
    self.config(mrc::pymrc::cast_from_pyobject(config));
}

}  // namespace morpheus
