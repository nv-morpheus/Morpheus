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

// TODO(dagardner): add /health & /info endpoints

#include "morpheus/utilities/rest_server.hpp"

#include <boost/asio.hpp>         // for dispatch
#include <boost/asio/ip/tcp.hpp>  // for acceptor, endpoint, socket,
#include <boost/beast/core.hpp>   // for bind_front_handler, error_code, flat_buffer, tcp_stream
#include <boost/beast/http.hpp>   // for read_async, request, response, verb, write_async
#include <glog/logging.h>         // for CHECK and LOG
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <thread>
#include <utility>  // for move

// loosely based on the following examples:
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/http/server/async/http_server_async.cpp
// https://www.boost.org/doc/libs/1_74_0/libs/beast/example/advanced/server/advanced_server.cpp

namespace {
namespace beast = boost::beast;  // from <boost/beast.hpp>
namespace http  = beast::http;   // from <boost/beast/http.hpp>
namespace net   = boost::asio;   // from <boost/asio.hpp>

// from <boost/asio/ip/tcp.hpp>
using tcp = boost::asio::ip::tcp;  // NOLINT(readability-identifier-naming)
using namespace std::literals::chrono_literals;

class Session : public std::enable_shared_from_this<Session>
{
  public:
    Session(tcp::socket&& socket,
            std::shared_ptr<morpheus::payload_parse_fn_t> payload_parse_fn,
            const std::string& url_endpoint,
            http::verb method,
            std::size_t max_payload_size,
            std::chrono::seconds timeout) :
      m_stream{std::move(socket)},
      m_payload_parse_fn{std::move(payload_parse_fn)},
      m_url_endpoint{url_endpoint},
      m_method{method},
      m_max_payload_size{max_payload_size},
      m_timeout{timeout},
      m_on_complete_cb{nullptr}
    {}

    ~Session() = default;

    void run()
    {
        net::dispatch(m_stream.get_executor(), beast::bind_front_handler(&Session::do_read, shared_from_this()));
    }

  private:
    void do_read()
    {
        m_parser = std::make_unique<http::request_parser<http::string_body>>();
        m_parser->body_limit(m_max_payload_size);
        m_stream.expires_after(m_timeout);

        http::async_read(
            m_stream, m_buffer, *m_parser, beast::bind_front_handler(&Session::on_read, shared_from_this()));
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

        // Release ownership of the parsed message and move it into the handle_request method
        handle_request(m_parser->release());
    }

    void handle_request(http::request<http::string_body>&& request)
    {
        DLOG(INFO) << "Received request: " << request.method() << " : " << request.target();
        m_response = std::make_unique<http::response<http::string_body>>();

        if (request.target() == m_url_endpoint && (request.method() == m_method))
        {
            std::string body{request.body()};
            auto parse_status = (*m_payload_parse_fn)(body);

            m_response->result(std::get<0>(parse_status));
            m_response->set(http::field::content_type, std::get<1>(parse_status));
            m_response->body() = std::get<2>(parse_status);
            m_on_complete_cb   = std::get<3>(parse_status);
        }
        else
        {
            m_response->result(http::status::not_found);
            m_response->set(http::field::content_type, "text/plain");
            m_response->body() = "not found";
        }

        try
        {
            DLOG(INFO) << "Response: " << m_response->result_int();
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

        m_parser.reset(nullptr);
        m_response.reset(nullptr);

        if (m_on_complete_cb)
        {
            try
            {
                std::thread([cb = std::move(this->m_on_complete_cb), ec]() { cb(ec); }).detach();
            } catch (const std::exception& e)
            {
                LOG(ERROR) << "Caught exception while calling on_complete callback: " << e.what();
            } catch (...)
            {
                LOG(ERROR) << "Caught unknown exception while calling on_complete callback";
            }

            m_on_complete_cb = nullptr;
        }

        do_read();
    }

    void do_close()
    {
        beast::error_code ec;
        m_stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }

    beast::tcp_stream m_stream;
    beast::flat_buffer m_buffer;
    std::shared_ptr<morpheus::payload_parse_fn_t> m_payload_parse_fn;
    const std::string& m_url_endpoint;
    http::verb m_method;
    std::size_t m_max_payload_size;
    std::chrono::seconds m_timeout;

    // The response, and parser are all reset for each incoming request
    std::unique_ptr<http::request_parser<http::string_body>> m_parser;
    std::unique_ptr<http::response<http::string_body>> m_response;
    morpheus::on_complete_cb_fn_t m_on_complete_cb;
};

class Listener : public std::enable_shared_from_this<Listener>
{
  public:
    Listener(std::shared_ptr<boost::asio::io_context> io_context,
             std::shared_ptr<morpheus::payload_parse_fn_t> payload_parse_fn,
             const std::string& bind_address,
             unsigned short port,
             const std::string& endpoint,
             http::verb method,
             std::size_t max_payload_size,
             std::chrono::seconds request_timeout) :
      m_io_context{std::move(io_context)},
      m_tcp_endpoint{net::ip::make_address(bind_address), port},
      m_acceptor{net::make_strand(*m_io_context)},
      m_payload_parse_fn{std::move(payload_parse_fn)},
      m_url_endpoint{endpoint},
      m_method{method},
      m_max_payload_size{max_payload_size},
      m_request_timeout{request_timeout}
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
            std::make_shared<Session>(
                std::move(socket), m_payload_parse_fn, m_url_endpoint, m_method, m_max_payload_size, m_request_timeout)
                ->run();
        }

        do_accept();
    }

    std::shared_ptr<boost::asio::io_context> m_io_context;
    tcp::endpoint m_tcp_endpoint;
    tcp::acceptor m_acceptor;

    std::shared_ptr<morpheus::payload_parse_fn_t> m_payload_parse_fn;
    const std::string& m_url_endpoint;
    http::verb m_method;
    std::size_t m_max_payload_size;
    std::chrono::seconds m_request_timeout;
};

}  // namespace

namespace morpheus {

RestServer::RestServer(payload_parse_fn_t payload_parse_fn,
                       std::string bind_address,
                       unsigned short port,
                       std::string endpoint,
                       std::string method,
                       unsigned short num_threads,
                       std::size_t max_payload_size,
                       std::chrono::seconds request_timeout) :
  m_payload_parse_fn(std::make_shared<payload_parse_fn_t>(std::move(payload_parse_fn))),
  m_bind_address(std::move(bind_address)),
  m_port(port),
  m_endpoint(std::move(endpoint)),
  m_method(http::string_to_verb(method)),
  m_num_threads(num_threads),
  m_request_timeout(request_timeout),
  m_max_payload_size(max_payload_size),
  m_io_context{nullptr},
  m_is_running{false}
{
    if (m_method == http::verb::unknown)
    {
        throw std::runtime_error("Invalid method: " + method);
    }

    if (m_num_threads == 0)
    {
        throw std::runtime_error("num_threads must be greater than 0");
    }
}

void RestServer::start_listener()
{
    DCHECK(m_io_context == nullptr) << "start_listener expects m_io_context to be null";

    // This function will block until the io context is shutdown, and should be called from the first worker thread
    DCHECK(m_listener_threads.size() == 1 && m_listener_threads[0].get_id() == std::this_thread::get_id())
        << "start_listener must be called from the first thread in m_listener_threads";

    m_io_context = std::make_shared<net::io_context>(m_num_threads);
    auto ioc     = m_io_context;  // ensure each thread gets its own copy including this one

    std::make_shared<Listener>(
        ioc, m_payload_parse_fn, m_bind_address, m_port, m_endpoint, m_method, m_max_payload_size, m_request_timeout)
        ->run();

    for (auto i = 1; i < m_num_threads; ++i)
    {
        m_listener_threads.emplace_back([ioc]() { ioc->run(); });
    }

    ioc->run();
}

void RestServer::start()
{
    CHECK(!m_is_running) << "RestServer is already running";

    try
    {
        DLOG(INFO) << "Starting RestServer on " << m_bind_address << ":" << m_port << " with " << m_num_threads
                   << " threads";
        m_is_running = true;
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
    m_is_running = false;
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
    return m_is_running;
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
                                                           unsigned short num_threads,
                                                           std::size_t max_payload_size,
                                                           int64_t request_timeout)
{
    payload_parse_fn_t payload_parse_fn = [py_parse_fn = std::move(py_parse_fn)](const std::string& payload) {
        pybind11::gil_scoped_acquire gil;
        auto py_payload = pybind11::str(payload);
        auto py_result  = pybind11::tuple(py_parse_fn(py_payload));
        on_complete_cb_fn_t cb_fn{nullptr};
        if (!py_result[3].is_none())
        {
            cb_fn = make_on_complete_wrapper(py_result[3]);
        }

        return std::make_tuple(pybind11::cast<unsigned>(py_result[0]),
                               pybind11::cast<std::string>(py_result[1]),
                               pybind11::cast<std::string>(py_result[2]),
                               std::move(cb_fn));
    };

    return std::make_shared<RestServer>(std::move(payload_parse_fn),
                                        std::move(bind_address),
                                        port,
                                        std::move(endpoint),
                                        std::move(method),
                                        num_threads,
                                        max_payload_size,
                                        std::chrono::seconds(request_timeout));
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

on_complete_cb_fn_t RestServerInterfaceProxy::make_on_complete_wrapper(pybind11::function py_on_complete_fn)
{
    return [py_cb_fn = std::move(py_on_complete_fn)](const beast::error_code& ec) {
        pybind11::gil_scoped_acquire gil;
        pybind11::print("on_complete_cb_fn_t lambda called");
        pybind11::bool_ has_error = false;
        pybind11::str error_msg;
        if (ec)
        {
            has_error = true;
            error_msg = ec.message();
        }

        pybind11::print("calling py_cb_fn");
        py_cb_fn(has_error, error_msg);
        pybind11::print("py_cb_fn lambda returned");
    };
}

}  // namespace morpheus
