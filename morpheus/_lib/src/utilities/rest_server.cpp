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

// loosely based on the following examples:
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/http/server/async/http_server_async.cpp
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/advanced/server/advanced_server.cpp

namespace {
namespace beast = boost::beast;          // from <boost/beast.hpp>
namespace http  = beast::http;           // from <boost/beast/http.hpp>
namespace net   = boost::asio;           // from <boost/asio.hpp>
using tcp       = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hppe>
using namespace std::literals::chrono_literals;

std::atomic<bool> g_is_running{false};

// std::make_shared<Session>(std::move(socket), m_payload_parse_fn, m_url_endpoint, m_method)->run();
class Session : public std::enable_shared_from_this<Session>
{
  public:
    Session(tcp::socket&& socket,
            std::shared_ptr<morpheus::payload_parse_fn_t> payload_parse_fn,
            const std::string& url_endpoint,
            http::verb method,
            std::chrono::seconds timeout = 30s /*TODO: expose timeout from RestServer & stage*/) :
      m_stream{std::move(socket)},
      m_payload_parse_fn{std::move(payload_parse_fn)},
      m_url_endpoint{url_endpoint},
      m_method{method},
      m_timeout{timeout}
    {}

    ~Session() = default;

    void run()
    {
        net::dispatch(m_stream.get_executor(), beast::bind_front_handler(&Session::do_read, shared_from_this()));
    }

  private:
    void do_read()
    {
        m_request = {};
        m_stream.expires_after(m_timeout);

        // TODO: replace with parser overload and make max payload size configurable
        http::async_read(
            m_stream, m_buffer, m_request, beast::bind_front_handler(&Session::on_read, shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec == http::error::end_of_stream)
        {
            return do_close();
        }

        if (ec)
        {
            LOG(ERROR) << "Error reading request: " << ec.message();
            return;
        }

        // move the request, this resets it for the next incoming request
        handle_request(std::move(m_request));
    }

    void handle_request(http::request<http::string_body>&& request)
    {
        DLOG(INFO) << "Received request: " << request.method() << " : " << request.target();
        m_response = std::make_unique<http::response<http::string_body>>();

        if (request.target() == m_url_endpoint && (request.method() == m_method))
        {
            std::string body{request.body()};
            auto parse_status = (*m_payload_parse_fn)(body);

            m_response->result(parse_status.first);
            m_response->body() = parse_status.second;
        }
        else
        {
            m_response->result(http::status::not_found);
            m_response->body() = "not found";
        }

        try
        {
            DLOG(INFO) << "Response: " << m_response->result_int() << " : " << m_response->body();
            m_response->set(http::field::content_type, "text/plain");
            m_response->keep_alive(request.keep_alive());
            m_response->prepare_payload();

            http::async_write(
                m_stream,
                *m_response,
                beast::bind_front_handler(&Session::on_write, shared_from_this(), m_response->need_eof()));
        } catch (const std::exception& e)
        {
            LOG(ERROR) << "Caught exception while writing response: " << e.what();
        }
    }

    void on_write(bool close, beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec)
        {
            LOG(ERROR) << "Error writing response: " << ec.message();
            return;
        }

        if (close)
        {
            return do_close();
        }

        m_response.reset(nullptr);
        do_read();
    }

    void do_close()
    {
        beast::error_code ec;
        m_stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }

    beast::tcp_stream m_stream;
    beast::flat_buffer m_buffer;
    http::request<http::string_body> m_request;
    std::unique_ptr<http::response<http::string_body>> m_response;
    std::shared_ptr<morpheus::payload_parse_fn_t> m_payload_parse_fn;
    const std::string& m_url_endpoint;
    http::verb m_method;
    std::chrono::seconds m_timeout;
};

class Listener : public std::enable_shared_from_this<Listener>
{
  public:
    Listener(std::shared_ptr<boost::asio::io_context> io_context,
             std::shared_ptr<morpheus::payload_parse_fn_t> payload_parse_fn,
             const std::string& bind_address,
             unsigned short port,
             const std::string& endpoint,
             http::verb method) :
      m_io_context{std::move(io_context)},
      m_tcp_endpoint{net::ip::make_address(bind_address), port},
      m_acceptor{net::make_strand(*m_io_context), m_tcp_endpoint},
      m_payload_parse_fn{std::move(payload_parse_fn)},
      m_url_endpoint{endpoint},
      m_method{method}
    {
        m_acceptor.open(m_tcp_endpoint.protocol());
        m_acceptor.set_option(net::socket_base::reuse_address(true));
        m_acceptor.bind(m_tcp_endpoint);
        m_acceptor.listen(net::socket_base::max_listen_connections);
    }

    ~Listener() = default;

    void run()
    {
        net::dispatch(m_acceptor.get_executor(),
                      beast::bind_front_handler(&Listener::do_accept, this->shared_from_this()));
    }

  private:
    void do_accept()
    {
        m_acceptor.async_accept(net::make_strand(*m_io_context),
                                beast::bind_front_handler(&Listener::on_accept, this->shared_from_this()));
    }

    void on_accept(beast::error_code ec, tcp::socket socket)
    {
        if (ec)
        {
            LOG(ERROR) << "Error accepting connection: " << ec.message();
        }
        else
        {
            std::make_shared<Session>(std::move(socket), m_payload_parse_fn, m_url_endpoint, m_method)->run();
        }

        do_accept();
    }

    std::shared_ptr<boost::asio::io_context> m_io_context;
    tcp::endpoint m_tcp_endpoint;
    tcp::acceptor m_acceptor;

    std::shared_ptr<morpheus::payload_parse_fn_t> m_payload_parse_fn;
    const std::string& m_url_endpoint;
    http::verb m_method;
};

}  // namespace

namespace morpheus {

RestServer::RestServer(payload_parse_fn_t payload_parse_fn,
                       std::string bind_address,
                       unsigned short port,
                       std::string endpoint,
                       std::string method,
                       unsigned short num_threads) :
  m_payload_parse_fn(std::make_shared<payload_parse_fn_t>(std::move(payload_parse_fn))),
  m_bind_address(std::move(bind_address)),
  m_port(port),
  m_endpoint(std::move(endpoint)),
  m_method(http::string_to_verb(method)),
  m_num_threads(num_threads),
  m_io_context{nullptr}
{
    if (m_method != http::verb::post && m_method != http::verb::put)
    {
        throw std::runtime_error("Invalid method: " + method);
    }
}

void RestServer::start_listener()
{
    DCHECK(!g_is_running) << "RestServer is already running";
    DCHECK(m_io_context == nullptr) << "start_listener expects m_io_context to be null";

    // This function will block until the io context is shutdown, and should be called from the first worker thread
    DCHECK(m_listener_threads.size() == 1 && m_listener_threads[0].get_id() == std::this_thread::get_id())
        << "start_listener must be called from the first thread in m_listener_threads";

    g_is_running = true;
    m_io_context = std::make_shared<net::io_context>(m_num_threads);
    auto ioc     = m_io_context;  // ensure each thread gets its own copy including this one

    std::make_shared<Listener>(ioc, m_payload_parse_fn, m_bind_address, m_port, m_endpoint, m_method)->run();

    for (auto i = 1; i < m_num_threads; ++i)
    {
        m_listener_threads.emplace_back([ioc]() { ioc->run(); });
    }

    ioc->run();
}

void RestServer::start()
{
    CHECK(!g_is_running) << "RestServer is already running";

    try
    {
        m_listener_threads.reserve(m_num_threads);
        m_listener_threads.emplace_back(std::thread(&RestServer::start_listener, this));
    } catch (const std::exception& e)
    {
        LOG(ERROR) << "Caught exception while starting rest server: " << e.what();
        stop();
    }
}

void RestServer::stop()
{
    g_is_running = false;
    if (m_io_context)
    {
        m_io_context->stop();
    }

    for (auto& t : m_listener_threads)
    {
        t.join();
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
                                                           std::string method,
                                                           unsigned short num_threads)
{
    payload_parse_fn_t payload_parse_fn = [py_parse_fn = std::move(py_parse_fn)](const std::string& payload) {
        pybind11::gil_scoped_acquire gil;
        auto py_payload = pybind11::str(payload);
        auto py_result  = py_parse_fn(py_payload);
        auto result     = pybind11::cast<parse_status_t>(py_result);
        return result;
    };

    return std::make_shared<RestServer>(std::move(payload_parse_fn),
                                        std::move(bind_address),
                                        port,
                                        std::move(endpoint),
                                        std::move(method),
                                        num_threads);
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
