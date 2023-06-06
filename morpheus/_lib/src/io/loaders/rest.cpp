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

std::shared_ptr<ControlMessage> RESTDataLoader::load(std::shared_ptr<ControlMessage> message, nlohmann::json task)
{
    VLOG(30) << "Called RESTDataLoader::load()";

    py::module_ mod_cudf;
    py::object dataframe;
    std::string strategy;
    // TODO(Devin) : error checking + improve robustness
    if (!task["queries"].is_array() or task.empty())
    {
        throw std::runtime_error("'REST Loader' control message specified no queries to load");
    }

    strategy = task.value("strategy", "aggregate");
    if (strategy != "aggregate")
    {
        throw std::runtime_error("Only 'aggregate' strategy is currently supported");
    }

    // Aggregate dataframes for each file
    {
        py::gil_scoped_acquire gil;

        auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
        mod_cudf           = cache_handle.get_module("cudf");
        dataframe          = py::none();
    }  // Release GIL

    auto queries = task["queries"];
    // TODO(Devin) : Migrate this to use the cudf::io interface
    for (auto& query : queries)
    {
        std::string endpoint = query.value("endpoint", "");
        if (endpoint.empty())
        {
            continue;
        }
        http::response<http::dynamic_body> response;
        response.clear();
        get_data_from_endpoint(endpoint, "/", "", response);
        {
            py::gil_scoped_acquire gil;
            auto current_df         = mod_cudf.attr("DataFrame")();
            
            std::string df_json_str = beast::buffers_to_string(response.body().data());
            std::cout << "content: " << df_json_str << std::endl;
            current_df              = mod_cudf.attr("read_json")(py::str(df_json_str));
            if (dataframe.is_none())
            {
                dataframe = current_df;
                continue;
            }
            if (strategy == "aggregate")
            {
                py::list args;
                args.attr("append")(dataframe);
                args.attr("append")(current_df);
                dataframe = mod_cudf.attr("concat")(args);
            }
        }  // Release GIL

        // Params in REST call not supported yet

        // auto params = query["params"];
        // for (auto& param : params)
        // {
        //     std::string param_str("?");
        //     for (auto& param_kv : param.items())
        //     {
        //         param_str += (std::string)param_kv.key() + "=" + (std::string)param_kv.value() + "&";
        //     }
        //     param_str.pop_back();
        //     response.clear();
        //     get_data_from_endpoint(endpoint, "/", param_str, response);
        //     {
        //         py::gil_scoped_acquire gil;
        //         auto current_df = mod_cudf.attr("DataFrame")();
        //         current_df = mod_cudf.attr("read_json")(beast::buffers_to_string(response.body().data()));
        //         if (dataframe.is_none())
        //         {
        //             dataframe = current_df;
        //             continue;
        //         }
        //         if (strategy == "aggregate") {
        //             py::list args;
        //             args.attr("append")(dataframe);
        //             args.attr("append")(current_df);
        //             dataframe = mod_cudf.attr("concat")(args);
        //         }
        //     } // Release GIL
        // }
    }
    
    message->payload(MessageMeta::create_from_python(std::move(dataframe)));
    return message;
}
}  // namespace morpheus