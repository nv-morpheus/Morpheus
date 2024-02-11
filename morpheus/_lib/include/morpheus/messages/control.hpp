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
     * If the key does not exist, the behavior depends on the fail_on_nonexist parameter.
     *
     * @param key A string indicating the metadata key.
     * @param fail_on_nonexist If true, throws an exception when the key does not exist.
     *                         If false, returns std::nullopt for non-existing keys.
     * @return An optional json object describing the metadata value if it exists.
     */
    std::optional<nlohmann::json> get_metadata(const std::string& key, bool fail_on_nonexist) const;

    std::vector<std::string> list_metadata() const;

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

    /**
     * @brief Sets a timestamp for a specific key within a given group.
     *
     * This method stores a nanosecond precision timestamp associated with a unique
     * identifier composed of a group and key. If the group::key already exists,
     * its timestamp will be updated to the new value.
     *
     * @param key The specific key for which the timestamp is to be set.
     * @param group The group to which the key belongs, aiding in namespace separation.
     * @param timestamp_ns The timestamp value in nanoseconds to be associated with the group::key.
     */
    void set_timestamp(const std::string& group, const std::string& key, std::size_t timestamp_ns);

    /**
     * @brief Retrieves the timestamp for a specific key within a given group.
     *
     * Attempts to find and return the timestamp associated with the specified group and key.
     * If the group::key combination does not exist, the method's behavior is determined by
     * the fail_if_nonexist flag.
     *
     * @param group The group to which the key belongs.
     * @param key The specific key for which the timestamp is requested.
     * @param fail_if_nonexist If true, the method throws an exception if the timestamp doesn't exist.
     *                         If false, returns std::nullopt for non-existing timestamps.
     * @return An optional containing the timestamp value if found, or std::nullopt otherwise.
     */
    std::optional<std::size_t> get_timestamp(const std::string& group, const std::string& key, bool fail_if_nonexist);

    /**
     * @brief Retrieves timestamps for all keys within a given group that match a regex pattern.
     *
     * Searches within the specified group for keys that match the provided regex filter and returns
     * a map of these keys and their associated timestamps. This allows for flexible queries within
     * the group's namespace based on pattern matching.
     *
     * @param group The group within which to search for keys.
     * @param regex_filter A regular expression pattern that keys must match to be included in the result.
     * @return A map containing the matching group::key combinations and their timestamps. The map will
     *         be empty if no matches are found.
     */
    std::map<std::string, std::size_t> get_timestamp(const std::string& group, const std::string& regex_filter);

  private:
    static const std::string s_config_schema;                          // NOLINT
    static std::map<std::string, ControlMessageType> s_task_type_map;  // NOLINT

    ControlMessageType m_cm_type{ControlMessageType::NONE};
    std::shared_ptr<MessageMeta> m_payload{nullptr};

    nlohmann::json m_tasks{};
    nlohmann::json m_config{};

    std::map<std::string, std::size_t> m_timestamps{};
};

struct ControlMessageProxy
{
    /**
     * @brief Creates a new ControlMessage instance from a configuration dictionary.
     * @param config A pybind11::dict representing the configuration for the ControlMessage.
     * @return A shared_ptr to a newly created ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> create(pybind11::dict& config);

    /**
     * @brief Creates a new ControlMessage instance as a copy of an existing one.
     * @param other A shared_ptr to another ControlMessage instance to copy.
     * @return A shared_ptr to the newly copied ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> create(std::shared_ptr<ControlMessage> other);

    /**
     * @brief Creates a deep copy of the ControlMessage instance.
     * @param self Reference to the underlying ControlMessage object.
     * @return A shared_ptr to the copied ControlMessage instance.
     */
    static std::shared_ptr<ControlMessage> copy(ControlMessage& self);

    /**
     * @brief Retrieves the configuration of the ControlMessage as a dictionary.
     * @param self Reference to the underlying ControlMessage object.
     * @return A pybind11::dict representing the ControlMessage's configuration.
     */
    static pybind11::dict config(ControlMessage& self);

    /**
     * @brief Updates the configuration of the ControlMessage from a dictionary.
     * @param self Reference to the underlying ControlMessage object.
     * @param config A pybind11::dict representing the new configuration.
     */
    static void config(ControlMessage& self, pybind11::dict& config);

    /**
     * @brief Adds a task to the ControlMessage.
     * @param self Reference to the underlying ControlMessage object.
     * @param type The type of the task to be added.
     * @param task A pybind11::dict representing the task to be added.
     */
    static void add_task(ControlMessage& self, const std::string& type, pybind11::dict& task);

    /**
     * @brief Removes and returns a task of the given type from the ControlMessage.
     * @param self Reference to the underlying ControlMessage object.
     * @param type The type of the task to be removed.
     * @return A pybind11::dict representing the removed task.
     */
    static pybind11::dict remove_task(ControlMessage& self, const std::string& type);

    /**
     * @brief Retrieves all tasks from the ControlMessage.
     * @param self Reference to the underlying ControlMessage object.
     * @return A pybind11::dict containing all tasks.
     */
    static pybind11::dict get_tasks(ControlMessage& self);

    /**
     * @brief Sets a metadata key-value pair.
     * @param self Reference to the underlying ControlMessage object.
     * @param key The key for the metadata entry.
     * @param value The value for the metadata entry, must be JSON serializable.
     */
    static void set_metadata(ControlMessage& self, const std::string& key, pybind11::object& value);

    /**
     * @brief Retrieves a metadata value by key, with an optional default value.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param key The key for the metadata entry. If not provided, retrieves all metadata.
     * @param default_value An optional default value to return if the key does not exist.
     * @return The value associated with the key, the default value if the key is not found, or all metadata if the key
     * is not provided.
     */
    static pybind11::object get_metadata(ControlMessage& self,
                                         std::optional<std::string> const& key,
                                         pybind11::object default_value);

    /**
     * @brief Lists all metadata keys of the ControlMessage.
     * @param self Reference to the underlying ControlMessage object.
     * @return A pybind11::list containing all metadata keys.
     */
    static pybind11::list list_metadata(ControlMessage& self);

    /**
     * @brief Sets a timestamp for a given key within a specified group.
     * @param self Reference to the underlying ControlMessage object.
     * @param key The key associated with the timestamp.
     * @param group The group the key belongs to.
     * @param timestamp_ns The timestamp value in nanoseconds.
     */
    static void set_timestamp(ControlMessage& self,
                              const std::string& group,
                              const std::string& key,
                              std::size_t timestamp_ns);

    /**
     * @brief Retrieves the timestamp for a specific key within a given group from the ControlMessage object.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param group The group to which the key belongs.
     * @param key The specific key for which the timestamp is requested.
     * @param fail_if_nonexist Determines the behavior when the requested timestamp does not exist.
     *                         If true, an exception is thrown. If false, py::none is returned.
     * @return The timestamp value if found, or py::none if not found and fail_if_nonexist is false.
     */
    static pybind11::object get_timestamp(ControlMessage& self,
                                          const std::string& group,
                                          const std::string& key,
                                          bool fail_if_nonexist);

    /**
     * @brief Retrieves timestamps for all keys within a given group that match a regex pattern from the ControlMessage
     * object.
     *
     * @param self Reference to the underlying ControlMessage object.
     * @param group The group within which to search for keys.
     * @param regex_filter The regex pattern that keys must match to be included in the result.
     * @return A Python dictionary of matching group::key combinations and their timestamps.
     */
    static pybind11::dict get_timestamp(ControlMessage& self,
                                        const std::string& group,
                                        const std::string& regex_filter);
};

#pragma GCC visibility pop
}  // namespace morpheus
