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

#pragma once

#include "morpheus/messages/meta.hpp"          // for MessageMeta
#include "morpheus/utilities/rest_server.hpp"  // for RestServer

#include <boost/fiber/buffered_channel.hpp>  // for buffered_channel
#include <cudf/io/types.hpp>                 // for table_with_metadata
#include <mrc/segment/builder.hpp>           // for segment::Builder
#include <mrc/segment/object.hpp>            // for segment::Object
#include <pymrc/node.hpp>                    // for PythonSource
#include <rxcpp/rx.hpp>                      // for subscriber

#include <chrono>   // for duration
#include <cstddef>  // for size_t
#include <cstdint>  // for int64_t
#include <memory>   // for shared_ptr & unique_ptr
#include <ratio>    // for std::milli
#include <string>   // for string & to_string

namespace morpheus {
using table_t         = std::unique_ptr<cudf::io::table_with_metadata>;
using request_queue_t = boost::fibers::buffered_channel<table_t>;

/****** Component public implementations *******************/
/****** RestSourceStage*************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)

class RestSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    RestSourceStage(std::string bind_address             = "127.0.0.1",
                    unsigned short port                  = 8080,
                    std::string endpoint                 = "/message",
                    std::string method                   = "POST",
                    float sleep_time                     = 0.1f,
                    long queue_timeout                   = 5,
                    std::size_t max_queue_size           = 1024,
                    unsigned short num_server_threads    = 1,
                    std::size_t max_payload_size         = DefaultMaxPayloadSize,
                    std::chrono::seconds request_timeout = std::chrono::seconds(30),
                    bool lines                           = false);
    ~RestSourceStage() override;

    void close();

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);

    std::chrono::duration<float, std::milli> m_sleep_time;
    std::chrono::duration<long> m_queue_timeout;
    std::unique_ptr<RestServer> m_server;
    request_queue_t m_queue;
};

/****** RestSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct RestSourceStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<RestSourceStage>> init(mrc::segment::Builder& builder,
                                                                       const std::string& name,
                                                                       std::string bind_address,
                                                                       unsigned short port,
                                                                       std::string endpoint,
                                                                       std::string method,
                                                                       float sleep_time,
                                                                       long queue_timeout,
                                                                       std::size_t max_queue_size,
                                                                       unsigned short num_server_threads,
                                                                       std::size_t max_payload_size,
                                                                       int64_t request_timeout,
                                                                       bool lines);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
