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

#include <functional>  // for function
#include <memory>      // for unique_ptr
#include <string>      // for string
#include <thread>      // for thread
#include <utility>     // for pair & move

// forward declare boost::beast::http::verb
namespace boost {
namespace beast {
namespace http {
enum class verb;
}  // namespace http
}  // namespace beast
}  // namespace boost

namespace morpheus {

using parse_status_t = std::pair<unsigned /*http status code*/, std::string /* http status message*/>;

// function that receives the post body and returns a status code and message
using payload_parse_fn_t = std::function<parse_status_t(const std::string& /* post body */)>;

class RestServer
{
  public:
    RestServer(payload_parse_fn_t payload_parse_fn,
               std::string bind_address = "127.0.0.1",
               unsigned short port      = 8080,
               std::string endpoint     = "/",
               std::string method       = "POST");
    ~RestServer();
    void start();
    void stop();
    bool is_running() const;

  private:
    std::string m_bind_address;
    unsigned short m_port;
    std::string m_endpoint;
    boost::beast::http::verb m_method;
    std::unique_ptr<std::thread> m_listener_thread;
    payload_parse_fn_t m_payload_parse_fn;
};

}  // namespace morpheus
