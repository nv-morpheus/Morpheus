/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/io/loaders/rest.hpp"

#include "morpheus/messages/control.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/system/error_code.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pymrc/utilities/object_cache.hpp>

#include <memory>
#include <ostream>
#include <stdexcept>

namespace py = pybind11;

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using tcp       = net::ip::tcp;

#define PORT "8081"

namespace morpheus {
RESTDataLoader::RESTDataLoader(nlohmann::json config) : Loader(config) {}

void extract_query_fields(nlohmann::basic_json<>& query,
                          std::string& method,
                          std::string& endpoint,
                          std::string& content_type,
                          std::string& body,
                          std::unordered_map<std::string, std::string> x_headers,
                          nlohmann::json& params)
{
    method = query.value("method", "GET");
    std::transform(method.begin(), method.end(), method.begin(), ::toupper);

    endpoint = query.value("endpoint", "");
    if (endpoint.empty())
    {
        throw std::runtime_error("'REST loader' receives query with empty endpoint");
    }

    content_type = query.value("content_type", "");
    body         = query.value("body", "");

    nlohmann::json x_headers_json = query.value("x-headers", nlohmann::json());
    for (auto& x_header_kv : x_headers_json.items())
    {
        x_headers.insert(std::make_pair((std::string)x_header_kv.key(), (std::string)x_header_kv.value()));
    }

    params = query.value("params", nlohmann::json());
}

http::verb get_http_verb(const std::string& method)
{
    http::verb verb;
    if (method == "GET")
    {
        verb = http::verb::get;
    }
    else if (method == "POST")
    {
        verb = http::verb::post;
    }
    else
    {
        std::string error_msg = "'REST Loader' receives method not supported: " + method;
        throw std::runtime_error(error_msg);
    }
    return verb;
}

http::request<http::string_body> construct_request(http::verb verb,
                                                   const std::string& target,
                                                   const std::string& query,
                                                   const std::string& host,
                                                   const std::string& method,
                                                   const std::string& content_type,
                                                   const std::string& body,
                                                   const std::unordered_map<std::string, std::string>& x_headers)
{
    http::request<http::string_body> request{verb, target + query, 11};
    request.set(http::field::host, host);
    request.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    if (method == "POST")
    {
        request.set(http::field::content_type, content_type);
        request.body() = body;
        request.prepare_payload();
    }
    for (auto& x_header : x_headers)
    {
        request.insert(x_header.first, x_header.second);
    }
    return request;
}

void try_get_data(beast::tcp_stream& stream,
                  const net::ip::basic_resolver_results<tcp>& endpoint_results,
                  http::request<http::string_body>& request,
                  http::response<http::dynamic_body>& response,
                  int max_retry,
                  int interval_milliseconds)
{
    while (max_retry-- > 0)
    {
        stream.connect(endpoint_results);
        http::write(stream, request);
        beast::flat_buffer buffer;
        http::read(stream, buffer, response);
        beast::error_code ec;
        stream.socket().shutdown(tcp::socket::shutdown_both, ec);
        if (ec && ec != beast::errc::not_connected)
        {
            throw beast::system_error{ec};
        }
        int status = response.result_int();
        // 503 Service Unavailable / 504 Gateway Timeout
        if (status != 503 && status != 504)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_milliseconds));
        interval_milliseconds *= 2;
    }
}

py::dict convert_raw_query_to_dict(const py::object& raw_query)
{
    py::dict params_dict;
    if (py::len(raw_query) > 0)
    {
        py::list split_by_amp = raw_query.attr("split")("&");

        for (auto& query_kv : split_by_amp)
        {
            py::list split_by_eq        = query_kv.attr("split")("=");
            params_dict[split_by_eq[0]] = split_by_eq[1];
        }
    }
    return params_dict;
}

void parse_and_format_url(std::string& endpoint,
                          std::unordered_map<std::string, std::string>& params,
                          std::string& query,
                          std::string& host,
                          std::string& target)
{
    py::gil_scoped_acquire gil;
    py::module_ urllib = py::module::import("urllib.parse");

    // Following the syntax specifications in RFC 1808, urlparse recognizes a netloc only if it is properly
    // introduced by ‘//’. Otherwise the input is presumed to be a relative URL and thus to start with a path
    // component. ref: https://docs.python.org/3/library/urllib.parse.html Workaround here is to prepend "//" if
    // endpoint does not already include one
    if (endpoint.find("//") == std::string::npos)
    {
        endpoint = "http://" + endpoint;
    }

    py::object result = urllib.attr("urlparse")(endpoint);
    py::str raw_query = result.attr("query");
    py::dict query_pydict;

    if (params.empty())
    {
        query_pydict = convert_raw_query_to_dict(py::object(result.attr("query")));
    }
    // if params exists, overrides query included in endpoint
    else
    {
        for (auto& param : params)
        {
            query_pydict[py::str(param.first)] = param.second;
        }
    }

    query  = py::len(query_pydict) == 0 ? "" : "?" + urllib.attr("urlencode")(query_pydict).cast<std::string>();
    host   = result.attr("hostname").is_none() ? "" : result.attr("hostname").cast<std::string>();
    target = urllib.attr("quote")(result.attr("path")).cast<std::string>();
    target = target.empty() ? "/" : target;
}

void get_data(http::response<http::dynamic_body>& response,
              const std::string& target,
              const std::string& query,
              const std::string& host,
              const std::string& method,
              const std::string& content_type,
              const std::string body,
              const std::unordered_map<std::string, std::string>& x_headers,
              int max_retry)
{
    net::io_context ioc;
    tcp::resolver resolver(ioc);
    beast::tcp_stream stream(ioc);
    auto const endpoint_results = resolver.resolve(host, PORT);
    auto verb                   = get_http_verb(method);
    auto request                = construct_request(verb, target, query, host, method, content_type, body, x_headers);

    try_get_data(stream, endpoint_results, request, response, max_retry, 1000);
}

void get_data_from_endpoint(http::response<http::dynamic_body>& response,
                            std::string& method,
                            std::string& endpoint,
                            std::unordered_map<std::string, std::string>& params,
                            std::string& content_type,
                            std::string& body,
                            std::unordered_map<std::string, std::string>& x_headers,
                            int max_retry)
{
    std::string query;
    std::string host;
    std::string target;
    parse_and_format_url(endpoint, params, query, host, target);
    get_data(response, target, query, host, method, content_type, body, x_headers, max_retry);
}

void create_dataframe_from_http_response(http::response<http::dynamic_body>& response,
                                         py::object& dataframe,
                                         py::module_& mod_cudf,
                                         const std::string& strategy)
{
    std::string df_json_str = beast::buffers_to_string(response.body().data());
    boost::algorithm::trim(df_json_str);

    // When calling cudf.read_json() with engine='cudf', it expects an array object as input.
    // The workaround here is to add square brackets if the original data is not represented as an array.
    // Submitted an issue to cudf: https://github.com/rapidsai/cudf/issues/13527.
    if (!(df_json_str.front() == '[' && df_json_str.back() == ']'))
    {
        df_json_str = "[" + df_json_str + "]";
    }

    {
        py::gil_scoped_acquire gil;
        auto current_df = mod_cudf.attr("DataFrame")();
        current_df      = mod_cudf.attr("read_json")(py::str(df_json_str), py::str("cudf"));
        if (dataframe.is_none())
        {
            dataframe = current_df;
        }
        else if (strategy == "aggregate")
        {
            py::list args;
            args.attr("append")(dataframe);
            args.attr("append")(current_df);
            dataframe = mod_cudf.attr("concat")(args);
        }
    }
}

void process_failures(const std::string& error_msg,
                      std::shared_ptr<ControlMessage> message,
                      bool processes_failures_as_errors)
{
    if (processes_failures_as_errors)
    {
        throw std::runtime_error(error_msg);
    }
    std::cout << "---------------------------------error_msg: " << error_msg << std::endl;
    message->set_metadata("failed", "true");
    message->set_metadata("failed reason", error_msg);
}

void process_python_failures(std::shared_ptr<ControlMessage> message, bool processes_failure_as_errors)
{
    // Retrieve the error message using Python C API
    PyObject* ptype;
    PyObject* pvalue;
    PyObject* ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    py::handle hType(ptype);
    py::handle hValue(pvalue);

    // Convert Python objects to strings
    py::str typeStr(hType);
    py::str valueStr(hValue);

    std::string error_msg = "Caught Python exception: " + std::string(typeStr) + ": " + std::string(valueStr);
    process_failures(error_msg, message, processes_failure_as_errors);
}

std::shared_ptr<ControlMessage> RESTDataLoader::load(std::shared_ptr<ControlMessage> message, nlohmann::json task)
{
    VLOG(30) << "Called RESTDataLoader::load()";

    auto conf                         = this->config();
    bool processes_failures_as_errors = conf.value("processes_failures_as_errors", false);
    int max_retry                     = conf.value("max_retry", 3);

    py::module_ mod_cudf;
    py::object dataframe;
    try
    {
        py::gil_scoped_acquire gil;

        dataframe          = py::none();
        auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
        mod_cudf           = cache_handle.get_module("cudf");
    } catch (py::error_already_set& e)
    {
        process_python_failures(message, processes_failures_as_errors);
        return message;
    }

    if (!task["queries"].is_array() or task.empty())
    {
        process_failures(
            "'REST Loader' control message specified no queries to load", message, processes_failures_as_errors);
        return message;
    }

    std::string strategy = task.value("strategy", "aggregate");
    if (strategy != "aggregate")
    {
        process_failures("Only 'aggregate' strategy is currently supported", message, processes_failures_as_errors);
        return message;
    }

    auto queries = task["queries"];

    for (auto& query : queries)
    {
        std::string method;
        std::string endpoint;
        std::string content_type;
        std::string body;
        std::unordered_map<std::string, std::string> x_headers;
        nlohmann::json params;
        try
        {
            extract_query_fields(query, method, endpoint, content_type, body, x_headers, params);
        } catch (const std::runtime_error& e)
        {
            process_failures(e.what(), message, processes_failures_as_errors);
            return message;
        }
        std::unordered_map<std::string, std::string> param_map;
        if (params.empty())
        {
            http::response<http::dynamic_body> response;
            try
            {
                get_data_from_endpoint(response, method, endpoint, param_map, content_type, body, x_headers, max_retry);
                create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
            } catch (const std::runtime_error& e)
            {
                process_failures(e.what(), message, processes_failures_as_errors);
                return message;
            }
        }
        else
        {
            for (auto& param : params)
            {
                for (auto& param_kv : param.items())
                {
                    param_map.insert(std::make_pair(param_kv.key(), param_kv.value()));
                }
                http::response<http::dynamic_body> response;
                try
                {
                    get_data_from_endpoint(
                        response, method, endpoint, param_map, content_type, body, x_headers, max_retry);
                    create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
                } catch (const std::runtime_error& e)
                {
                    process_failures(e.what(), message, processes_failures_as_errors);
                    return message;
                }
            }
        }
    }
    try
    {
        py::gil_scoped_acquire gil;
        message->payload(MessageMeta::create_from_python(std::move(dataframe)));
    } catch (py::error_already_set& e)
    {
        process_python_failures(message, processes_failures_as_errors);
        return message;
    }
    return message;
}
}  // namespace morpheus