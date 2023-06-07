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
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace morpheus {
RESTDataLoader::RESTDataLoader(nlohmann::json config) : Loader(config) {}

void get_data_from_endpoint(const std::string& host,
                            const std::string& target,
                            const std::string& params,
                            http::response<http::dynamic_body>& res)
{
    try
    {
        net::io_context ioc;
        tcp::resolver resolver(ioc);
        beast::tcp_stream stream(ioc);
        auto const results = resolver.resolve(host, "8081");
        stream.connect(results);
        http::request<http::string_body> req{http::verb::get, target + params, 11};
        req.set(http::field::host, host);
        req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
        http::write(stream, req);
        beast::flat_buffer buffer;
        http::read(stream, buffer, res);
        beast::error_code ec;
        stream.socket().shutdown(tcp::socket::shutdown_both, ec);
        if (ec && ec != beast::errc::not_connected)
        {
            throw beast::system_error{ec};
        }
    } catch (std::exception const& e)
    {
        // TODO: VLOG here?
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void create_dataframe_from_http_response(http::response<http::dynamic_body>& response, py::object& dataframe, py::module_& mod_cudf, const std::string& strategy) {
        
        std::string df_json_str = beast::buffers_to_string(response.body().data());
        boost::algorithm::trim(df_json_str);

        // When calling cudf.read_json() with engine='cudf', it expects an array object as input.
        // The workaround here is to add square brackets if the original data is not represented as an array.
        // Submitted an issue to cudf: https://github.com/rapidsai/cudf/issues/13527.
        if (!(df_json_str.front() == '[' && df_json_str.back() == ']')) {
            df_json_str = "[" + df_json_str + "]";
        }
        std::cout << "content: " << df_json_str << std::endl;
        auto current_df         = mod_cudf.attr("DataFrame")();
        current_df = mod_cudf.attr("read_json")(py::str(df_json_str), py::str("cudf"));
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

    py::object dataframe = py::none();
    auto queries = task["queries"];
    // TODO(Devin) : Migrate this to use the cudf::io interface
    for (auto& query : queries)
    {
        std::string endpoint = query.value("endpoint", "");
        if (endpoint.empty())
        {
            continue;
        }
        nlohmann::json params = query.value("params", nlohmann::json());
        if (params.empty()) {
            http::response<http::dynamic_body> response;
            get_data_from_endpoint(endpoint, "/", "", response);
            create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
        }
        else {
            for (auto& param : params)
            {
                std::string param_str("?");
                for (auto& param_kv : param.items())
                {
                    param_str += (std::string)param_kv.key() + "=" + (std::string)param_kv.value() + "&";
                }
                param_str.pop_back();
                http::response<http::dynamic_body> response;
                get_data_from_endpoint(endpoint, "/", param_str, response);
                create_dataframe_from_http_response(response, dataframe, mod_cudf, strategy);
            }
        }
    }

    message->payload(MessageMeta::create_from_python(std::move(dataframe)));
    // py::gil_scoped_release release;
    return message;
}
}  // namespace morpheus