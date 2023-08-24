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
#include "morpheus/utilities/http_server.hpp"  // for HttpServer

#include <boost/fiber/buffered_channel.hpp>  // for buffered_channel
#include <boost/fiber/context.hpp>           // for context
#include <boost/fiber/future/future.hpp>
#include <cudf/io/types.hpp>               // for table_with_metadata
#include <mrc/node/rx_sink_base.hpp>       // for RxSinkBase
#include <mrc/node/rx_source_base.hpp>     // for RxSourceBase
#include <mrc/node/source_properties.hpp>  // for channel::Status, SourceProperties<>::source_type_t
#include <mrc/segment/builder.hpp>         // for segment::Builder
#include <mrc/segment/object.hpp>          // for segment::Object
#include <mrc/types.hpp>                   // for SegmentAddress
#include <pymrc/node.hpp>                  // for PythonSource
#include <rxcpp/rx.hpp>                    // for subscriber

#include <chrono>   // for duration
#include <cstddef>  // for size_t
#include <cstdint>  // for int64_t
#include <map>
#include <memory>  // for shared_ptr & unique_ptr
#include <ratio>   // for std::milli
#include <string>  // for string & to_string
#include <vector>
// IWYU thinks we're using thread::operator<<
// IWYU pragma: no_include <thread>

namespace morpheus {
using table_t         = std::unique_ptr<cudf::io::table_with_metadata>;
using request_queue_t = boost::fibers::buffered_channel<table_t>;

/****** Component public implementations *******************/
/****** HttpServerSourceStage *************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

#pragma GCC visibility push(default)

// TODO(dagardner): optionally add headers to the dataframe

class HttpServerSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    HttpServerSourceStage(std::string bind_address             = "127.0.0.1",
                          unsigned short port                  = 8080,
                          std::string endpoint                 = "/message",
                          std::string method                   = "POST",
                          unsigned accept_status               = 201,
                          float sleep_time                     = 0.1f,
                          long queue_timeout                   = 5,
                          std::size_t max_queue_size           = 1024,
                          unsigned short num_server_threads    = 1,
                          std::size_t max_payload_size         = DefaultMaxPayloadSize,
                          std::chrono::seconds request_timeout = std::chrono::seconds(30),
                          bool lines                           = false,
                          std::size_t stop_after               = 0);
    ~HttpServerSourceStage() override;

    void close();

  private:
    subscriber_fn_t build();
    void source_generator(rxcpp::subscriber<source_type_t> subscriber);

    std::chrono::duration<float, std::milli> m_sleep_time;
    std::chrono::duration<long> m_queue_timeout;
    std::unique_ptr<HttpServer> m_server;
    request_queue_t m_queue;
    std::size_t m_stop_after;
    std::size_t m_records_emitted;
};

/****** HttpServerSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct HttpServerSourceStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<HttpServerSourceStage>> init(mrc::segment::Builder& builder,
                                                                             const std::string& name,
                                                                             std::string bind_address,
                                                                             unsigned short port,
                                                                             std::string endpoint,
                                                                             std::string method,
                                                                             unsigned accept_status,
                                                                             float sleep_time,
                                                                             long queue_timeout,
                                                                             std::size_t max_queue_size,
                                                                             unsigned short num_server_threads,
                                                                             std::size_t max_payload_size,
                                                                             int64_t request_timeout,
                                                                             bool lines,
                                                                             std::size_t stop_after);
};
#pragma GCC visibility pop
/** @} */  // end of group
}  // namespace morpheus
