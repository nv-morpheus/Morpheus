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

MessageControl::MessageControl(const nlohmann::json& message) : m_message(message) {}

const nlohmann::json& MessageControl::message() const
{
    return m_message;
}

void MessageControl::message(const nlohmann::json& message)
{
    m_message = message;
}

/*** Proxy Implementations ***/

std::shared_ptr<MessageControl> ControlMessageProxy::create(py::dict& message)
{
    return std::make_shared<MessageControl>(mrc::pymrc::cast_from_pyobject(message));
}

py::dict ControlMessageProxy::message(MessageControl& self)
{
    auto dict = mrc::pymrc::cast_from_json(self.message());

    return dict;
}

void ControlMessageProxy::message(MessageControl& self, py::dict& message)
{
    self.message(mrc::pymrc::cast_from_pyobject(message));
}

}  // namespace morpheus
