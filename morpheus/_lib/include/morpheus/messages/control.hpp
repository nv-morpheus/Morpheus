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
enum class ControlMessageType
{
    CUSTOM,
    DATA,
    INFERENCE,
    NONE,
    TRAINING
};

class MessageControl
{
  public:
    MessageControl();
    MessageControl(const nlohmann::json& config);
    MessageControl(const MessageControl& other);  // NO payload copy

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
     * @brief Add a task to the control message
     * @param task
     * @param type
     */
    void add_task(const std::string& task_type, const nlohmann::json& task);

    /**
     * @brief Check if a task of a given type exists
     * @param type
     * @return
     */
    bool has_task(const std::string& task_type) const;

    /**
     * @brief Get a task of the given type
     * @param type
     * @return
     */
    const nlohmann::json pop_task(const std::string& task_type);

    /**
     * @brief Add a metadata key-value pair to the control message
     * @param key
     * @param value
     */
    void set_metadata(const std::string& key, const nlohmann::json& value);

    /**
     * @brief Check if a metadata key exists
     * @param key
     * @return
     */
    bool has_metadata(const std::string& key) const;

    /**
     * @brief Get the metadata value for a given key
     * @param key
     * @return
     */
    const nlohmann::json get_metadata(const std::string& key) const;

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

    /**
     * @brief Get the type of the task
     * @return ControlMessageType
     */
    ControlMessageType task_type() const;

    /**
     * @brief Set the task type for the control message
     * @param task_type
     * @return
     */
    void task_type(ControlMessageType task_type);

  private:
    static const std::string s_config_schema;  // NOLINT

    ControlMessageType m_cm_type{ControlMessageType::NONE};
    std::shared_ptr<MessageMeta> m_payload{nullptr};
    std::map<std::string, ControlMessageType> m_task_type_map{{"inference", ControlMessageType::INFERENCE},
                                                              {"training", ControlMessageType::TRAINING}};

    nlohmann::json m_tasks{};
    nlohmann::json m_config{};
};

struct ControlMessageProxy
{
    static std::shared_ptr<MessageControl> create(pybind11::dict& config);
    static std::shared_ptr<MessageControl> create(std::shared_ptr<MessageControl> other);

    static std::shared_ptr<MessageControl> copy(MessageControl& self);

    static pybind11::dict config(MessageControl& self);

    // Required for proxy conversion of json -> dict in python
    static void config(MessageControl& self, pybind11::dict& config);

    static void add_task(MessageControl& self, const std::string& type, pybind11::dict& task);
    static pybind11::dict pop_task(MessageControl& self, const std::string& type);

    /**
     * @brief Set a metadata key-value pair -- value must be json serializable
     * @param self
     * @param key
     * @param value
     */
    static void set_metadata(MessageControl& self, const std::string& key, pybind11::object& value);
    static pybind11::object get_metadata(MessageControl& self, const std::string& key);
};

#pragma GCC visibility pop
}  // namespace morpheus
