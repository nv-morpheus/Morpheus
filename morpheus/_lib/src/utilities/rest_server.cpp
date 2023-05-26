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

// TODO:
// use fiber buffer chann between server & source
// source passes a lambda to server, and capures the fiber buffer chan
// If fiber buffer chan is full, return 503
// Use an async listener

// add /health & /info endpoints

#include "morpheus/utilities/rest_server.hpp"

#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/http/verb.hpp>
#include <boost/beast/version.hpp>
#include <boost/config.hpp>
#include <glog/logging.h>  // for CHECK and LOG

#include <atomic>  // for atomic
#include <memory>
#include <string>
#include <utility>  // for move

namespace {
namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http  = beast::http;           // from <boost/beast/http.hpp>
namespace net   = boost::asio;           // from <boost/asio.hpp>
using tcp       = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hppe>

std::atomic<bool> g_is_running{false};

void listener(std::shared_ptr<morpheus::RequestQueue> queue,
              const std::string& bind_address,
              unsigned short port,
              const std::string& endpoint,
              http::verb method)
{
    // loosely based on
    // https://www.boost.org/doc/libs/1_74_0/libs/beast/example/http/server/sync/http_server_sync.cpp

    // TODO Look into :
    // * https://www.boost.org/doc/libs/1_74_0/libs/beast/example/http/server/fast/http_server_fast.cpp
    // * https://www.boost.org/doc/libs/1_74_0/libs/beast/example/advanced/server/advanced_server.cpp

    auto const address = net::ip::make_address(bind_address);
    net::io_context ioc{1};  // TODO concurrency
    tcp::acceptor acceptor{ioc, {address, port}};

    while (g_is_running)
    {
        tcp::socket socket{ioc};
        acceptor.accept(socket);  // blocking TODO: interupt this when g_is_running is set to false

        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        http::response<http::string_body> res;

        try
        {
            // TODO this needs to be a nested loop as any bytes read past the first requrest will be
            // stored in the buffer and read in the next iteration
            http::read(socket, buffer, req);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Caught exception while reading request: " << e.what();
            continue;
        }

        DLOG(INFO) << "Received request: " << req.method() << " : " << req.target();
        if (req.target() == endpoint && (req.method() == method))
        {
            std::string body{req.body()};

            // TODO: Consider parsing to a cudf table here and then passing that to the queue
            // that would allow us to return an error code if the parsing fails, and parsing would be
            // performed in the worker thread instead of the stage's thread
            queue->push(std::move(body));
            res.result(http::status::created);  // TODO make this a config option
            res.set(http::field::content_type, "text/plain");
            res.body() = "";
            res.prepare_payload();
        }
        else
        {
            res.result(http::status::not_found);
            res.set(http::field::content_type, "text/plain");
            res.body() = "not found";
            res.prepare_payload();
        }

        try
        {
            http::write(socket, res);
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Caught exception while writing response: " << e.what();
            continue;
        }

        beast::error_code ec;
        socket.shutdown(tcp::socket::shutdown_send, ec);
    }
}

}  // namespace

namespace morpheus {

RestServer::RestServer(std::string bind_address, unsigned short port, std::string endpoint, std::string method) :
  m_bind_address(std::move(bind_address)),
  m_port(port),
  m_endpoint(std::move(endpoint)),
  m_method(http::string_to_verb(method)),
  m_queue(std::make_shared<RequestQueue>(1024))
{
    if (m_method != http::verb::post && m_method != http::verb::put)
    {
        throw std::runtime_error("Invalid method: " + method);
    }
}

void RestServer::start()
{
    CHECK(!g_is_running) << "RestServer is already running";

    try
    {
        g_is_running      = true;
        m_listener_thread = std::make_unique<std::thread>(
            [this]() { listener(m_queue, m_bind_address, m_port, m_endpoint, m_method); });
    } catch (const std::exception& e)
    {
        g_is_running = false;
        LOG(ERROR) << "Caught exception while starting rest server: " << e.what();
    }
}

void RestServer::stop()
{
    g_is_running = false;
    if (m_listener_thread)
    {
        m_listener_thread->join();
    }
}

bool RestServer::is_running() const
{
    return g_is_running;
}

std::shared_ptr<RequestQueue> RestServer::get_queue() const
{
    return m_queue;
}

RestServer::~RestServer()
{
    try
    {
        stop();
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while stopping rest server: " << e.what();
    }
}
}  // namespace morpheus
