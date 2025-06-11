/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/io/loaders/payload.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <ostream>
#include <utility>

namespace morpheus {
PayloadDataLoader::PayloadDataLoader(nlohmann::json config) : Loader(config) {}

std::shared_ptr<ControlMessage> PayloadDataLoader::load(std::shared_ptr<ControlMessage> message, nlohmann::json task)
{
    VLOG(30) << "Called PayloadDataLoader::load()";
    return std::move(message);
}
}  // namespace morpheus