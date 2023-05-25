/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/stages/rest_source.hpp"

#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"
#include "pymrc/node.hpp"

#include "morpheus/io/deserializers.hpp"
#include "morpheus/objects/table_info.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <boost/lockfree/queue.hpp>
#include <cudf/io/json.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <mrc/segment/builder.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>  // for str_attr_accessor
#include <pybind11/pytypes.h>   // for pybind11::int_

#include <functional>
#include <memory>
#include <sstream>
#include <thread>  // for std::this_thread::sleep_for
#include <utility>

namespace morpheus {
// Component public implementations
// ************ RestSourceStage ************* //
RestSourceStage::RestSourceStage(std::string bind_address,
                                 unsigned short port,
                                 std::string endpoint,
                                 float sleep_time) :
  PythonSource(build()),
  m_bind_address{std::move(bind_address)},
  m_port{port},
  m_endpoint{std::move(endpoint)},
  m_sleep_time{sleep_time},
  m_server{std::make_unique<RestServer>(m_bind_address, m_port, m_endpoint)},
  m_queue{m_server->get_queue()}
{}

RestSourceStage::subscriber_fn_t RestSourceStage::build()
{
    return [this](rxcpp::subscriber<source_type_t> subscriber) -> void {
        try
        {
            m_server->start();
            this->source_generator(subscriber);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Encountered error while listening for incoming REST requests: " << e.what() << std::endl;
            subscriber.on_error(std::make_exception_ptr(e));
            return;
        }
        subscriber.on_completed();
    };
}

void RestSourceStage::source_generator(rxcpp::subscriber<RestSourceStage::source_type_t> subscriber)
{
    while (subscriber.is_subscribed() && m_server->is_running())
    {
        std::string payload;
        if (m_queue->pop(payload))
        {
            try
            {
                cudf::io::source_info source{payload.c_str(), payload.size()};
                auto options = cudf::io::json_reader_options::builder(source).lines(true);
                auto table   = cudf::io::read_json(options.build());
                auto message = MessageMeta::create_from_cpp(std::move(table), 0);
                subscriber.on_next(std::move(message));
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Error occurred converting REST payload to Dataframe: " << e.what() << "\n" << payload;
            }
        }
        else
        {
            // Sleep when there are no messages
            std::this_thread::sleep_for(m_sleep_time);
        }
    }
}

// ************ RestSourceStageInterfaceProxy ************ //
std::shared_ptr<mrc::segment::Object<RestSourceStage>> RestSourceStageInterfaceProxy::init(
    mrc::segment::Builder& builder,
    const std::string& name,
    std::string bind_address,
    unsigned short port,
    std::string endpoint,
    float sleep_time)
{
    return builder.construct_object<RestSourceStage>(
        name, std::move(bind_address), port, std::move(endpoint), sleep_time);
}
}  // namespace morpheus
