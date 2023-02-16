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

#include "morpheus/modules/data_loader_module.hpp"

#include "morpheus/io/loaders/all.hpp"
#include "morpheus/messages/meta.hpp"

#include <mrc/modules/segment_modules.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/utils/type_utils.hpp>
#include <nlohmann/json.hpp>

#include <string>

using namespace mrc::modules;
using nlohmann::json;

namespace morpheus {

DataLoaderModule::DataLoaderModule(std::string module_name) : SegmentModule(module_name) {}

DataLoaderModule::DataLoaderModule(std::string module_name, nlohmann::json _config) :
  SegmentModule(std::move(module_name), std::move(_config))
{}

void DataLoaderModule::initialize(mrc::segment::Builder& builder)
{
    // TODO(Devin): Modularize loader lookups, and standardize this a bit more
    if (config().contains("loaders") and config()["loaders"].size() > 0)
    {
        auto loader_list = config()["loaders"];
        for (json::iterator it = loader_list.begin(); it != loader_list.end(); ++it)
        {
            if (*it == "file")
            {
                m_data_loader.add_loader("file", std::make_unique<FileDataLoader>());
            }
            else if (*it == "grpc")
            {
                m_data_loader.add_loader("grpc", std::make_unique<GRPCDataLoader>());
            }
            else if (*it == "payload")
            {
                m_data_loader.add_loader("payload", std::make_unique<PayloadDataLoader>());
            }
            else if (*it == "rest")
            {
                m_data_loader.add_loader("rest", std::make_unique<RESTDataLoader>());
            }
            else
            {
                throw std::runtime_error("Unknown or unsupported loader type: " + (*it).dump());
            }
        }
    }
    else
    {
        LOG(WARNING) << "No loaders specified in config";
    }

    auto loader_node = builder.make_node<std::shared_ptr<MessageControl>, std::shared_ptr<MessageMeta>>(
        "input", rxcpp::operators::map([this](std::shared_ptr<MessageControl> control_message) {
            return m_data_loader.load(*control_message);
        }));

    register_input_port("input", loader_node);
    register_output_port("output", loader_node);
}

std::string DataLoaderModule::module_type_name() const
{
    return std::string(::mrc::boost_type_name<type_t>());
}
}  // namespace morpheus
