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

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>

#include <memory>  // for shared_ptr, unique_ptr
#include <string>  // for string
#include <thread>  // for thread
#include <vector>

// forward declare boost::beast::http::verb
namespace boost {
namespace beast {
namespace http {
enum class verb;
}  // namespace http
}  // namespace beast
}  // namespace boost

namespace morpheus {

using RequestQueue = boost::fibers::buffered_channel<std::string>;

class RestServer
{
  public:
    RestServer(std::string bind_address = "127.0.0.1",
               unsigned short port      = 8080,
               std::string endpoint     = "/",
               std::string method       = "POST");
    ~RestServer();
    void start();
    void stop();
    bool is_running() const;
    std::shared_ptr<RequestQueue> get_queue() const;

  private:
    std::string m_bind_address;
    unsigned short m_port;
    std::string m_endpoint;
    boost::beast::http::verb m_method;
    std::unique_ptr<std::thread> m_listener_thread;
    std::shared_ptr<RequestQueue> m_queue;
};

}  // namespace morpheus
