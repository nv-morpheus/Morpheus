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
    MessageControl(const nlohmann::json& config);

    /**
     * @brief Set the config object
     * @param config
     */
    void config(const nlohmann::json& config);

    /**
     * @brief Get the config object
     * @return
     */
    const nlohmann::json& config() const;

    /**
     * @brief Set the payload object
     * @param payload
     */
    void payload(const std::shared_ptr<MessageMeta>& payload);

    /**
     * @brief Get the payload object
     * @return Shared pointer to the message payload
     */
    std::shared_ptr<MessageMeta> payload();

  private:
    static const std::string s_config_schema;  // NOLINT

    std::shared_ptr<MessageMeta> m_payload{nullptr};
    nlohmann::json m_config{};
};

struct ControlMessageProxy
{
    static std::shared_ptr<MessageControl> create(pybind11::dict& config);

    static pybind11::dict config(MessageControl& self);
    static void config(MessageControl& self, pybind11::dict& config);
};

#pragma GCC visibility pop
}  // namespace morpheus
