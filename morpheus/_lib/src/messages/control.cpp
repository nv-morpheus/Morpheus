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

MessageControl::MessageControl(const nlohmann::json& config) : m_config(config) {}

const nlohmann::json& MessageControl::config() const
{
    return m_config;
}

void MessageControl::config(const nlohmann::json& config)
{
    m_config = config;
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

py::dict ControlMessageProxy::config(MessageControl& self)
{
    auto dict = mrc::pymrc::cast_from_json(self.config());

    return dict;
}

void ControlMessageProxy::config(MessageControl& self, py::dict& config)
{
    self.config(mrc::pymrc::cast_from_pyobject(config));
}

}  // namespace morpheus
