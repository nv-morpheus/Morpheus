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

#include <mrc/utils/type_utils.hpp>
#include <pybind11/pybind11.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace morpheus {
#pragma GCC visibility push(default)
template <typename ObjectReturnTypeT>
class FactoryRegistry
{
  public:
    static std::shared_ptr<ObjectReturnTypeT> get_constructor(const std::string& name)
    {
        if (m_object_constructors.count(name) == 0)
        {
            throw std::runtime_error("Unknown data loader: " + name);
        }
        return m_object_constructors[name]();
    }

    static void register_constructor(const std::string& name,
                                     const std::function<std::shared_ptr<ObjectReturnTypeT>()>& loader_fn)
    {
        if (m_object_constructors.count(name) > 0)
        {
            throw std::runtime_error("Duplicate data loader registration: " + name);
        }
        m_object_constructors[name] = loader_fn;
    }

    static void unregister_constructor(const std::string& name, bool optional = false)
    {
        if (m_object_constructors.count(name) == 0)
        {
            if (optional)
            {
                return;
            }
            throw std::runtime_error("Unknown data loader: " + name);
        }
        m_object_constructors.erase(name);
    }

  private:
    static std::map<std::string, std::function<std::shared_ptr<Loader>()>> m_object_constructors;
};

// TODO(Devin): this shouldn't be templated, and should be specific to Loader
template <typename ObjectReturnTypeT>
struct FactoryRegistryProxy
{
    template <typename ReturnTypeT, typename... ArgsT>
    static void register_proxy_constructor(const std::string& name,
                                           std::function<ReturnTypeT(ArgsT...)> proxy_constructor);

    static void register_factory_cleanup_fn(const std::string& name)
    {
        {
            auto at_exit = pybind11::module_::import("atexit");
            at_exit.attr("register")(pybind11::cpp_function([name]() {
                VLOG(2) << "(atexit) Unregistering loader: " << name;

                // Try unregister -- ignore if already unregistered
                FactoryRegistry<Loader>::unregister_constructor(name, true);
            }));
        }
    }
};
#pragma GCC visibility pop

}  // namespace morpheus