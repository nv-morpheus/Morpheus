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

#include <boost/asio.hpp>  // for io_context
#include <boost/asio/io_context.hpp>
#include <pybind11/pytypes.h>  // for pybind11::function

#include <functional>  // for function
#include <memory>      // for shared_ptr & unique_ptr
#include <string>      // for string
#include <thread>      // for thread
#include <utility>     // for pair & move
#include <vector>      // for vector

// forward declare boost::beast::http::verb
namespace boost {
namespace beast {
namespace http {
enum class verb;
}  // namespace http
}  // namespace beast
}  // namespace boost

namespace morpheus {
/**
 * @addtogroup objects
 * @{
 * @file
 */

#pragma GCC visibility push(default)
/**
 * @brief A pair of unsigned and string, where the unsigned is the HTTP status code and the string
 *        is the HTTP status message.
 */
using parse_status_t = std::pair<unsigned /*http status code*/, std::string /* http status message*/>;

/**
 * @brief A function that receives the post body and returns a status code and message.
 *
 * @details The function is expected to return a pair of unsigned and string, where the unsigned is
 *          the HTTP status code and the string is the HTTP status message.
 */
using payload_parse_fn_t = std::function<parse_status_t(const std::string& /* post body */)>;

/**
 * @brief A simple REST server that listens for POST requests on a given endpoint.
 *
 * @details The server is started in a separate thread and will call the provided payload_parse_fn_t
 *          function when a POST request is received. The payload_parse_fn_t function is expected to
 *          return a pair of unsigned and string, where the unsigned is the HTTP status code and the
 *          string is the HTTP status message.
 *
 * @param payload_parse_fn The function that will be called when a POST request is received.
 * @param bind_address The address to bind the server to.
 * @param port The port to bind the server to.
 * @param endpoint The endpoint to listen for POST requests on.
 * @param method The HTTP method to listen for.
 * @param num_threads The number of threads to use for the server.
 */
class RestServer
{
  public:
    RestServer(payload_parse_fn_t payload_parse_fn,
               std::string bind_address   = "127.0.0.1",
               unsigned short port        = 8080,
               std::string endpoint       = "/message",
               std::string method         = "POST",
               unsigned short num_threads = 1);
    ~RestServer();
    void start();
    void stop();
    bool is_running() const;

  private:
    void start_listener();

    std::string m_bind_address;
    unsigned short m_port;
    std::string m_endpoint;
    boost::beast::http::verb m_method;
    unsigned short m_num_threads;
    std::vector<std::thread> m_listener_threads;
    std::shared_ptr<boost::asio::io_context> m_io_context;
    std::shared_ptr<payload_parse_fn_t> m_payload_parse_fn;
};

/****** RestServerInterfaceProxy *************************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct RestServerInterfaceProxy
{
    static std::shared_ptr<RestServer> init(pybind11::function py_parse_fn,
                                            std::string bind_address,
                                            unsigned short port,
                                            std::string endpoint,
                                            std::string method,
                                            unsigned short num_threads);
    static void start(RestServer& self);
    static void stop(RestServer& self);
    static bool is_running(const RestServer& self);
};
}  // namespace morpheus
