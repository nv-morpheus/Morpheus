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

ControlMessage::ControlMessage(const nlohmann::json& message) : m_message(message) {}

ControlMessage::ControlMessageType ControlMessage::type() const
{
    return m_type;
}

const nlohmann::json& ControlMessage::message() const
{
    return m_message;
}

std::shared_ptr<ControlMessage> ControlMessageProxy::create(py::dict& message)
{
    return std::make_shared<ControlMessage>(mrc::pymrc::cast_from_pyobject(message));
}

}  // namespace morpheus
