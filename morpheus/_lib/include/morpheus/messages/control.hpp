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

#pragma once

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>

#include <memory>

namespace morpheus {
class MessageMeta;
#pragma GCC visibility push(default)

class MessageControl
{
  public:
    MessageControl() = default;
    MessageControl(const nlohmann::json& message);

    /**
     * @brief Set the message object
     * @param message
     */
    void message(const nlohmann::json& message);

    /**
     *
     * @return
     */
    const nlohmann::json& message() const;

    void payload(const std::shared_ptr<MessageMeta>& payload);

    std::shared_ptr<MessageMeta> payload();

  private:
    std::shared_ptr<MessageMeta> m_payload{nullptr};
    nlohmann::json m_message{};
};

struct ControlMessageProxy
{
    static std::shared_ptr<MessageControl> create(pybind11::dict& message);

    static pybind11::dict message(MessageControl& self);
    static void message(MessageControl& self, pybind11::dict& message);
};

#pragma GCC visibility pop
}  // namespace morpheus
