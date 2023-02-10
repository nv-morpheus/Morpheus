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

namespace morpheus {

std::shared_ptr<MessageMeta> DataLoader::load(const MessageControl& control_message)
{
    auto payload = control_message.message();

    if (payload.contains("loader_id"))
    {
        auto loader_id = payload["loader_id"].get<std::string>();
        auto loader    = m_loaders.find(loader_id);
        if (loader != m_loaders.end())
        {
            return loader->second->load(control_message);
        }
    }

    // TODO(Devin): Testing. Remove this.
    return std::shared_ptr<MessageMeta>(nullptr);

    throw std::runtime_error("No loader registered for message: " + control_message.message().dump());
}

void DataLoader::register_loader(const std::string& loader_id, std::unique_ptr<Loader> loader)
{
    if (m_loaders.find(loader_id) != m_loaders.end())
    {
        throw std::runtime_error("Loader already registered with id: " + loader_id);
    }

    m_loaders[loader_id] = std::move(loader);
}

void DataLoader::remove_loader(const std::string& loader_id)
{
    if (m_loaders.find(loader_id) == m_loaders.end())
    {
        throw std::runtime_error("Loader not registered with id: " + loader_id);
    }

    m_loaders.erase(loader_id);
}
}  // namespace morpheus
