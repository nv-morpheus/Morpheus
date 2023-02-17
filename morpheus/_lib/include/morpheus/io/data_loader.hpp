/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <map>
#include <memory>

namespace morpheus {

#pragma GCC visibility push(default)
class Loader
{
  public:
    virtual ~Loader() = default;

    virtual std::shared_ptr<MessageMeta> load(MessageControl& message) = 0;
};

class DataLoader
{
  public:
    DataLoader();
    ~DataLoader() = default;

    /**
     * @brief Load data described by a control message
     * @param control_message
     * @return
     */
    std::shared_ptr<MessageMeta> load(MessageControl& control_message);

    /**
     * @brief Register a loader instance with the data loader
     * @param loader_id
     * @param loader
     * @param overwrite
     */
    void add_loader(const std::string& loader_id, std::shared_ptr<Loader> loader, bool overwrite = true);

    /**
     * @brief Remove a loader instance from the data loader
     * @param loader_id
     * @param throw_if_not_found
     */
    void remove_loader(const std::string& loader_id, bool throw_if_not_found = true);

  private:
    std::map<std::string, std::shared_ptr<Loader>> m_loaders{};
};
#pragma GCC visibility pop
}  // namespace morpheus