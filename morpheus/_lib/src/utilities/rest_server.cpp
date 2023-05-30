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
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <atomic>  // for atomic
#include <memory>

namespace {
namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http  = beast::http;           // from <boost/beast/http.hpp>
namespace net   = boost::asio;           // from <boost/asio.hpp>
using tcp       = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hppe>

std::atomic<bool> g_is_running{false};

void listener(morpheus::payload_parse_fn_t* payload_parse_fn,
              const std::string& bind_address,
              unsigned short port,
              const std::string& endpoint,
              http::verb method)
{
    DCHECK_NOTNULL(payload_parse_fn);
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
            auto parse_status = (*payload_parse_fn)(body);

            res.result(parse_status.first);
            res.body() = parse_status.second;
        }
        else
        {
            res.result(http::status::not_found);
            res.body() = "not found";
        }

        try
        {
            DLOG(INFO) << "Response: " << res.result_int() << " : " << res.body();
            res.set(http::field::content_type, "text/plain");
            res.prepare_payload();
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

RestServer::RestServer(payload_parse_fn_t payload_parse_fn,
                       std::string bind_address,
                       unsigned short port,
                       std::string endpoint,
                       std::string method) :
  m_payload_parse_fn(std::move(payload_parse_fn)),
  m_bind_address(std::move(bind_address)),
  m_port(port),
  m_endpoint(std::move(endpoint)),
  m_method(http::string_to_verb(method))
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
            [this]() { listener(&m_payload_parse_fn, m_bind_address, m_port, m_endpoint, m_method); });
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

/****** RestServerInterfaceProxy *************************/
std::shared_ptr<RestServer> RestServerInterfaceProxy::init(pybind11::function py_parse_fn,
                                                           std::string bind_address,
                                                           unsigned short port,
                                                           std::string endpoint,
                                                           std::string method)
{
    payload_parse_fn_t payload_parse_fn = [py_parse_fn = std::move(py_parse_fn)](const std::string& payload) {
        pybind11::gil_scoped_acquire gil;
        auto py_payload = pybind11::str(payload);
        auto py_result  = py_parse_fn(py_payload);
        auto result     = pybind11::cast<parse_status_t>(py_result);
        return result;
    };

    return std::make_shared<RestServer>(
        std::move(payload_parse_fn), std::move(bind_address), port, std::move(endpoint), std::move(method));
}

void RestServerInterfaceProxy::start(RestServer& self)
{
    self.start();
}

void RestServerInterfaceProxy::stop(RestServer& self)
{
    self.stop();
}

bool RestServerInterfaceProxy::is_running(const RestServer& self)
{
    return self.is_running();
}

}  // namespace morpheus
