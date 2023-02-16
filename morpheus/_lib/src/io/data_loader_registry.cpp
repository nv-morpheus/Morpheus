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
#include "morpheus/io/loaders/lambda.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/objects/factory_registry.hpp"

namespace morpheus {
template <>
std::map<std::string, std::function<std::shared_ptr<Loader>()>> FactoryRegistry<Loader>::m_object_constructors{};

template class FactoryRegistry<Loader>;

template <>
template <>
void FactoryRegistryProxy<Loader>::register_proxy_constructor(
    const std::string& name,
    std::function<std::shared_ptr<MessageMeta>(MessageControl& control_message)> proxy_constructor)
{
    FactoryRegistry<Loader>::register_constructor(name, [proxy_constructor]() {
        return std::make_shared<LambdaLoader>([proxy_constructor](MessageControl& control_message) {
            return std::move(proxy_constructor(control_message));
        });
    });

    register_factory_cleanup_fn(name);
}

}  // namespace morpheus