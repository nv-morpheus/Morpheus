/*
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

#include "morpheus/messages/memory/tensor_memory.hpp"
#include "morpheus/messages/meta.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/pytypes.h>

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace morpheus {
class MessageMeta;

#pragma GCC visibility push(default)
enum class ControlMessageType
{
    NONE,
    INFERENCE,
    TRAINING
};

// class PayloadManager
// {
//   public:
//     /**
//      * @brief Get the tensor object identified by `name`
//      *
//      * @param name
//      * @return TensorObject&
//      * @throws std::runtime_error If no tensor matching `name` exists
//      */
//     TensorObject& get_tensor(const std::string& name)
//     {
//         return m_tensors->get_tensor(name);
//     }

//     /**
//      * @brief Get the tensor object identified by `name`
//      *
//      * @param name
//      * @return const TensorObject&
//      * @throws std::runtime_error If no tensor matching `name` exists
//      */
//     const TensorObject& get_tensor(const std::string& name) const
//     {
//         return m_tensors->get_tensor(name);
//     }

//     /**
//      * @brief Set the tensor object identified by `name`
//      *
//      * @param name
//      * @param tensor
//      * @throws std::length_error If the number of rows in `tensor` does not match `count`.
//      */
//     void set_tensor(const std::string& name, TensorObject&& tensor)
//     {
//         m_tensors->set_tensor(name, std::move(tensor));
//     }

//     /**
//      * @brief Get a reference to the internal tensors map
//      *
//      * @return const TensorMap&
//      */
//     const TensorMap& get_tensors() const
//     {
//         return m_tensors->get_tensors();
//     }

//     /**
//      * @brief Set the tensors object
//      *
//      * @param tensors
//      * @throws std::length_error If the number of rows in the `tensors` do not match `count`.
//      */
//     void set_tensors(TensorMap&& tensors)
//     {
//         m_tensors->set_tensors(std::move(tensors));
//     }

//         /**
//      * @brief Get the tensor object identified by `name`
//      *
//      * @param name
//      * @return TensorObject&
//      * @throws std::runtime_error If no tensor matching `name` exists
//      */
//     TensorObject& get_column(const std::string& name)
//     {
//         return m_tensors->get_tensor(name);
//     }

//     /**
//      * @brief Get the tensor object identified by `name`
//      *
//      * @param name
//      * @return const TensorObject&
//      * @throws std::runtime_error If no tensor matching `name` exists
//      */
//     const TensorObject& get_column(const std::string& name) const
//     {
//         return m_tensors->get_tensor(name);
//     }

//     /**
//      * @brief Set the tensor object identified by `name`
//      *
//      * @param name
//      * @param tensor
//      * @throws std::length_error If the number of rows in `tensor` does not match `count`.
//      */
//     void set_column(const std::string& name, TensorObject&& tensor)
//     {
//         m_tensors->set_tensor(name, std::move(tensor));
//     }

//     /**
//      * @brief Get a reference to the internal tensors map
//      *
//      * @return const TensorMap&
//      */
//     TableInfo get_columns() const
//     {
//         return m_df->get_info();
//     }

//     /**
//      * @brief Set the tensors object
//      *
//      * @param tensors
//      * @throws std::length_error If the number of rows in the `tensors` do not match `count`.
//      */
//     void set_columns(TableInfo&& tensors)
//     {
//         m_tensors->set_tensors(std::move(tensors));
//     }

//   private:
//     std::shared_ptr<MessageMeta> m_df;
//     std::shared_ptr<TensorMemory> m_tensors;
// };

/**
 * @brief Class representing a control message for coordinating data processing tasks.
 *
 * This class contains configuration information, task definitions, and metadata, as well as a
 * pointer to an associated message payload. It provides methods for accessing and modifying these
 * elements of the control message.
 */
class ControlMessage
{
  public:
    ControlMessage();
    ControlMessage(const nlohmann::json& config);
    ControlMessage(const ControlMessage& other);  // Copies config and metadata, but not payload

    /**
     * @brief Set the configuration object for the control message.
     * @param config A json object containing configuration information.
     */
    void config(const nlohmann::json& config);

    /**
     * @brief Get the configuration object for the control message.
     * @return A const reference to the json object containing configuration information.
     */
    const nlohmann::json& config() const;

    /**
     * @brief Add a task of the given type to the control message.
     * @param task_type A string indicating the type of the task.
     * @param task A json object describing the task.
     */
    void add_task(const std::string& task_type, const nlohmann::json& task);

    /**
     * @brief Check if a task of the given type exists in the control message.
     * @param task_type A string indicating the type of the task.
     * @return True if a task of the given type exists, false otherwise.
     */
    bool has_task(const std::string& task_type) const;

    /**
     * @brief Remove and return a task of the given type from the control message.
     * @param task_type A string indicating the type of the task.
     * @return A json object describing the task.
     */
    const nlohmann::json remove_task(const std::string& task_type);

    /**
     * @brief Get the tasks for the control message.
     */
    const nlohmann::json& get_tasks() const;

    /**
     * @brief Add a key-value pair to the metadata for the control message.
     * @param key A string key for the metadata value.
     * @param value A json object describing the metadata value.
     */
    void set_metadata(const std::string& key, const nlohmann::json& value);

    /**
     * @brief Check if a metadata key exists in the control message.
     * @param key A string indicating the metadata key.
     * @return True if the metadata key exists, false otherwise.
     */
    bool has_metadata(const std::string& key) const;

    /**
     * @brief Get the metadata for the control message.
     */
    const nlohmann::json& get_metadata() const;

    /**
     * @brief Get the metadata value for the given key from the control message.
     * @param key A string indicating the metadata key.
     * @return A json object describing the metadata value.
     */
    const nlohmann::json get_metadata(const std::string& key) const;

    /**
     * @brief Get all metadata keys for the control message.
     * @return A json object containing all metadata keys and values.
     */
    const nlohmann::json list_metadata() const;

    /**
     * @brief Set the payload object for the control message.
     * @param payload
     * A shared pointer to the message payload.
     */
    std::shared_ptr<MessageMeta> payload();

    /**
     * @brief Set the payload object
     * @param payload
     */
    void payload(const std::shared_ptr<MessageMeta>& payload);

    /**
     * @brief Get the type of task associated with the control message.
     * @return An enum value indicating the task type.
     */
    ControlMessageType task_type();

    /**
     * @brief Set the task type for the control message
     * @param task_type
     * @return
     */
    void task_type(ControlMessageType task_type);

  private:
    static const std::string s_config_schema;                          // NOLINT
    static std::map<std::string, ControlMessageType> s_task_type_map;  // NOLINT

    ControlMessageType m_cm_type{ControlMessageType::NONE};
    std::shared_ptr<MessageMeta> m_payload{nullptr};

    nlohmann::json m_tasks{};
    nlohmann::json m_config{};
};

struct ControlMessageProxy
{
    static std::shared_ptr<ControlMessage> create(pybind11::dict& config);
    static std::shared_ptr<ControlMessage> create(std::shared_ptr<ControlMessage> other);

    static std::shared_ptr<ControlMessage> copy(ControlMessage& self);

    static pybind11::dict config(ControlMessage& self);

    // Required for proxy conversion of json -> dict in python
    static void config(ControlMessage& self, pybind11::dict& config);

    static void add_task(ControlMessage& self, const std::string& type, pybind11::dict& task);
    static pybind11::dict remove_task(ControlMessage& self, const std::string& type);
    static pybind11::dict get_tasks(ControlMessage& self);

    /**
     * @brief Set a metadata key-value pair -- value must be json serializable
     * @param self
     * @param key
     * @param value
     */
    static void set_metadata(ControlMessage& self, const std::string& key, pybind11::object& value);
    static pybind11::object get_metadata(ControlMessage& self, std::optional<std::string> const& key);

    static pybind11::dict list_metadata(ControlMessage& self);
};

#pragma GCC visibility pop
}  // namespace morpheus
