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

#include "morpheus/io/data_loader.hpp"

#include "morpheus/messages/control.hpp"

#include <ios>
#include <iostream>

namespace morpheus {

DataLoader::DataLoader() : m_loaders{} {}

Loader::Loader(nlohmann::json config)
{
    m_config = std::move(config);
}

nlohmann::json Loader::config() const
{
    return m_config;
}

std::shared_ptr<MessageMeta> Loader::payload(std::shared_ptr<MessageControl> message)
{
    return std::move(message->payload());
}

std::shared_ptr<MessageControl> Loader::load(std::shared_ptr<MessageControl> message, nlohmann::json task)
{
    return std::move(message);
}

std::shared_ptr<MessageControl> DataLoader::load(std::shared_ptr<MessageControl> control_message)
{
    while (control_message->has_task("load"))
    {
        auto task      = control_message->pop_task("load");
        auto loader_id = task["loader_id"];

        auto loader = m_loaders.find(loader_id);
        if (loader != m_loaders.end())
        {
            VLOG(5) << "Loading data using loader: " << loader_id
                    << " for message: " << control_message->config().dump() << std::endl;

            loader->second->load(control_message, task);
        }
        else
        {
            throw std::runtime_error("Attempt to load using an unknown or unregistered data loader: " +
                                     loader_id.get<std::string>());
        }
    }

    return std::move(control_message);
}

void DataLoader::add_loader(const std::string& loader_id, std::shared_ptr<Loader> loader, bool overwrite)
{
    if (!overwrite and m_loaders.find(loader_id) != m_loaders.end())
    {
        throw std::runtime_error("Loader already registered with id: " + loader_id);
    }

    VLOG(2) << "Registering data loader: " << loader_id << std::endl;

    m_loaders[loader_id] = std::move(loader);
}

void DataLoader::remove_loader(const std::string& loader_id, bool throw_if_not_found)
{
    if (m_loaders.find(loader_id) == m_loaders.end())
    {
        if (throw_if_not_found)
        {
            throw std::runtime_error("Loader not registered with id: " + loader_id);
        }

        return;
    }

    VLOG(2) << "Removing data loader: " << loader_id << std::endl;

    m_loaders.erase(loader_id);
}
}  // namespace morpheus
