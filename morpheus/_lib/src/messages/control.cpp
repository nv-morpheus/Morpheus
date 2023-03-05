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

const std::string MessageControl::s_config_schema = R"(
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ControlMessage",
    "type": "object",
    "required": ["tasks"],
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "properties"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["load", "inference", "training"]
                    },
                    "properties": {
                        "type": "object",
                        "allOf": [
                            {
                                "if": {
                                    "properties": {
                                        "type": { "const": "load" },
                                        "loader_id": { "const": "file" }
                                    }
                                },
                                "then": {
                                    "required": ["loader_id", "strategy", "files"],
                                    "properties": {
                                        "loader_id": { "type": "string", "enum": ["file"] },
                                        "strategy": { "type": "string" },
                                        "files": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "required": ["path", "type"],
                                                "properties": {
                                                    "path": { "type": "string" },
                                                    "type": { "type": "string" }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            {
                                "if": {
                                    "properties": {
                                        "type": { "const": "load" },
                                        "loader_id": { "const": "file_list" }
                                    }
                                },
                                "then": {
                                    "required": ["loader_id", "strategy", "directories"],
                                    "properties": {
                                        "loader_id": { "type": "string", "enum": ["file_list"] },
                                        "strategy": { "type": "string" },
                                        "directories": {
                                            "type": "array",
                                            "items": { "type": "string" }
                                        }
                                    }
                                }
                            },
                            {
                                "if": {
                                    "properties": {
                                        "type": { "enum": ["inference", "training"] }
                                    }
                                },
                                "then": {
                                    "properties": {
                                        "params": { "type": "object" }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    }
}
)";

MessageControl::MessageControl() :
  m_task_config({{"tasks", nlohmann::json::array()}, {"metadata", nlohmann::json::object()}})
{}

MessageControl::MessageControl(const nlohmann::json& _config) :
  m_task_config({{"tasks", nlohmann::json::array()}, {"metadata", nlohmann::json::object()}})
{
    config(_config);
}

MessageControl::MessageControl(const MessageControl& other)
{
    m_task_config = other.m_task_config;
    m_task_count  = other.m_task_count;
}

const nlohmann::json& MessageControl::config() const
{
    return m_task_config;
}

void MessageControl::add_task(const std::string& task_type, const nlohmann::json& task)
{
    // TODO(Devin) Schema check
    VLOG(20) << "Adding task of type " << task_type << " to control message" << task.dump(4);
    m_task_count[task_type] += 1;
    m_task_config["tasks"].push_back({{"type", task_type}, {"properties", task}});
}

bool MessageControl::has_task(const std::string& task_type) const
{
    return m_task_count.contains(task_type) and m_task_count.at(task_type) > 0;
}

void MessageControl::set_metadata(const std::string& key, const nlohmann::json& value)
{
    if (m_task_config["metadata"].contains(key))
    {
        LOG(WARNING) << "Overwriting metadata key " << key << " with value " << value;
    }

    m_task_config["metadata"][key] = value;
}

bool MessageControl::has_metadata(const std::string& key) const
{
    return m_task_config["metadata"].contains(key);
}

const nlohmann::json MessageControl::get_metadata(const std::string& key) const
{
    return m_task_config["metadata"].at(key);
}

const nlohmann::json MessageControl::pop_task(const std::string& task_type)
{
    auto& tasks = m_task_config["tasks"];
    for (auto it = tasks.begin(); it != tasks.end(); ++it)
    {
        if (it->at("type") == task_type)
        {
            auto task = *it;
            tasks.erase(it);
            m_task_count[task_type] -= 1;

            return task["properties"];
        }
    }

    throw std::runtime_error("No tasks of type " + task_type + " found");
}

void MessageControl::config(const nlohmann::json& config)
{
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

std::shared_ptr<MessageMeta> MessageControl::payload()
{
    // auto temp = std::move(m_payload);
    //  TODO(Devin): Decide if we copy or steal the payload
    //  m_payload = nullptr;

    return m_payload;
}

void MessageControl::payload(const std::shared_ptr<MessageMeta>& payload)
{
    m_payload = payload;
}

/*** Proxy Implementations ***/

std::shared_ptr<MessageControl> ControlMessageProxy::create(py::dict& config)
{
    return std::make_shared<MessageControl>(mrc::pymrc::cast_from_pyobject(config));
}

std::shared_ptr<MessageControl> ControlMessageProxy::create(std::shared_ptr<MessageControl> other)
{
    return std::make_shared<MessageControl>(*other);
}

std::shared_ptr<MessageControl> ControlMessageProxy::copy(MessageControl& self)
{
    return std::make_shared<MessageControl>(self);
}

void ControlMessageProxy::add_task(MessageControl& self, const std::string& task_type, py::dict& task)
{
    self.add_task(task_type, mrc::pymrc::cast_from_pyobject(task));
}

py::dict ControlMessageProxy::pop_task(MessageControl& self, const std::string& task_type)
{
    auto task = self.pop_task(task_type);

    return mrc::pymrc::cast_from_json(task);
}

py::dict ControlMessageProxy::config(MessageControl& self)
{
    auto dict = mrc::pymrc::cast_from_json(self.config());

    return dict;
}

py::object ControlMessageProxy::get_metadata(MessageControl& self, const std::string& key)
{
    auto dict = mrc::pymrc::cast_from_json(self.get_metadata(key));

    return dict;
}

void ControlMessageProxy::set_metadata(MessageControl& self, const std::string& key, pybind11::object& value)
{
    self.set_metadata(key, mrc::pymrc::cast_from_pyobject(value));
}

void ControlMessageProxy::config(MessageControl& self, py::dict& config)
{
    self.config(mrc::pymrc::cast_from_pyobject(config));
}

}  // namespace morpheus
