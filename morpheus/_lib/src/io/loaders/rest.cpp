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

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
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

namespace morpheus {
RESTDataLoader::RESTDataLoader(nlohmann::json config) : Loader(config) {}

void get_data_from_endpoint(const std::string& method,
                            const std::string& endpoint,
                            const std::string& params,
                            const std::string& content_type,
                            const std::string& body,
                            const std::unordered_map<std::string, std::string>& x_headers_map,
                            int max_retry,
                            http::response<http::dynamic_body>& res)
{
    py::module_ urllib = py::module::import("urllib.parse");
    std::string ep(endpoint);
    py::object ep_pystr = py::str(ep);

    // Following the syntax specifications in RFC 1808, urlparse recognizes a netloc only if it is properly introduced
    // by ‘//’. Otherwise the input is presumed to be a relative URL and thus to start with a path component. ref:
    // https://docs.python.org/3/library/urllib.parse.html Workaround here is to prepend "//" if endpoint does not
    // already include one
    if (ep_pystr.attr("find")("//").cast<int>() == -1)
    {
        ep = "//" + ep;
    }

    py::object result  = urllib.attr("urlparse")(ep);
    std::string host   = result.attr("hostname").is_none() ? "" : result.attr("hostname").cast<std::string>();
    std::string target = result.attr("path").is_none() ? "" : result.attr("path").cast<std::string>();
    if (target.empty())
    {
        target = "/";
    }
    std::string query = result.attr("query").is_none() ? "" : result.attr("query").cast<std::string>();
    if (!params.empty())
    {
        query = params;
    }

    net::io_context ioc;
    tcp::resolver resolver(ioc);
    beast::tcp_stream stream(ioc);
    try
    {
        auto const results = resolver.resolve(host, "8081");
        
        http::verb verb;
        if (method == "GET")
        {
            verb = http::verb::get;
        }
        else if (method == "POST")
        {
            verb = http::verb::post;
        }
        http::request<http::string_body> req{verb, target + query, 11};
        req.set(http::field::host, host);
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        if (method == "POST")
        {
            req.set(http::field::content_type, content_type);
            req.body() = body;
            req.prepare_payload();
        }

        for (auto& x_header : x_headers_map) {
            req.insert(x_header.first, x_header.second);
        }

        int interval_milliseconds = 1000;
        while (max_retry-- > 0) {
            std::cout << "max_retry: " << max_retry << std::endl; 
            stream.connect(results);
            http::write(stream, req);
            beast::flat_buffer buffer;
            http::read(stream, buffer, res);
            beast::error_code ec;
            stream.socket().shutdown(tcp::socket::shutdown_both, ec);
            if (ec && ec != beast::errc::not_connected)
            {
                throw beast::system_error{ec};
            }
            int status = res.result_int();
            // 503 Service Unavailable 504 Gateway Timeout
            if (status != 503 && status != 504) {
                break;
            }
            std::cout << "interval: " << interval_milliseconds << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_milliseconds));
            interval_milliseconds *= 2;
        }
       
    } catch (std::exception const& e)
    {
        // TODO: VLOG here?
        std::cerr << "Error: " << e.what() << std::endl;
    }

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
    std::cout << "content: " << df_json_str << std::endl;
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

std::shared_ptr<ControlMessage> RESTDataLoader::load(std::shared_ptr<ControlMessage> message, nlohmann::json task)
{
    VLOG(30) << "Called RESTDataLoader::load()";

    // Aggregate dataframes for each file
    py::gil_scoped_acquire gil;
    py::module_ mod_cudf;

    auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
    mod_cudf           = cache_handle.get_module("cudf");

    // TODO(Devin) : error checking + improve robustness
    if (!task["queries"].is_array() or task.empty())
    {
        throw std::runtime_error("'REST Loader' control message specified no queries to load");
    }

    std::string strategy = task.value("strategy", "aggregate");
    if (strategy != "aggregate")
    {
        throw std::runtime_error("Only 'aggregate' strategy is currently supported");
    }

    auto conf = this->config();
    int max_retry = conf.value("max_retry", 3);

    py::object dataframe = py::none();
    auto queries         = task["queries"];
    // TODO(Devin) : Migrate this to use the cudf::io interface
    for (auto& query : queries)
    {
        std::string method = query.value("method", "GET");
        std::transform(method.begin(), method.end(), method.begin(), ::toupper);
        std::string endpoint = query.value("endpoint", "");
        if (endpoint.empty())
        {
            continue;
        }
        std::string content_type = query.value("content_type", "");
        std::string body         = query.value("body", "");
        nlohmann::json x_headers = query.value("x-headers", nlohmann::json());
        std::unordered_map<std::string, std::string> x_headers_map;
        for (auto& x_header_kv : x_headers.items()) {
            x_headers_map.insert(std::make_pair((std::string)x_header_kv.key(), (std::string)x_header_kv.value()));
        }
        nlohmann::json params    = query.value("params", nlohmann::json());
        if (params.empty())
        {
            std::string param_str("");
            http::response<http::dynamic_body> response;
            get_data_from_endpoint(method, endpoint, param_str, content_type, body, x_headers_map, max_retry, response);
            create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
        }
        else
        {
            for (auto& param : params)
            {
                std::string param_str("?");
                for (auto& param_kv : param.items())
                {
                    param_str += (std::string)param_kv.key() + "=" + (std::string)param_kv.value() + "&";
                }
                param_str.pop_back();
                http::response<http::dynamic_body> response;
                get_data_from_endpoint(method, endpoint, param_str, content_type, body, x_headers_map, max_retry, response);
                create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
            }
        }
    }

    message->payload(MessageMeta::create_from_python(std::move(dataframe)));
    // py::gil_scoped_release release;
    return message;
}
}  // namespace morpheus