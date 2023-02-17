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

#include "morpheus/io/loaders/file.hpp"

#include "morpheus/messages/control.hpp"
#include "morpheus/messages/meta.hpp"

#include <boost/filesystem.hpp>
#include <pybind11/pybind11.h>
#include <pymrc/utilities/object_cache.hpp>

#include <fstream>
#include <memory>

namespace {}

namespace morpheus {
std::shared_ptr<MessageControl> FileDataLoader::load(std::shared_ptr<MessageControl> message)
{
    namespace py = pybind11;
    VLOG(30) << "Called FileDataLoader::load()";

    // Aggregate dataframes for each file
    py::gil_scoped_acquire gil;
    py::module_ mod_cudf;

    auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
    mod_cudf           = cache_handle.get_module("cudf");

    // TODO(Devin) : error checking + improve robustness
    auto config = message->config();
    if (!config.contains("files"))
    {
        throw std::runtime_error("'File Loader' control message specified no files to load");
    }

    // TODO(Devin) : Migrate this to use the cudf::io interface
    std::string strategy = config.value("strategy", "aggregate");
    if (strategy != "aggregate")
    {
        throw std::runtime_error("Only 'aggregate' strategy is currently supported");
    }

    auto files           = config["files"];
    py::object dataframe = py::none();
    for (auto& file : files)
    {
        boost::filesystem::path path(file.value("path", ""));
        std::string extension = file.value("type", path.extension().string());
        // Remove the leading period
        if (!extension.empty() && extension[0] == '.')
        {
            extension = extension.substr(1);
        }
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        VLOG(5) << "Loading file: " << file.dump(2);

        // TODO(Devin): Any extensions missing?
        auto current_df = mod_cudf.attr("DataFrame")();
        if (extension == "csv")
        {
            current_df = mod_cudf.attr("read_csv")(path.string());
        }
        else if (extension == "parquet")
        {
            current_df = mod_cudf.attr("read_parquet")(path.string());
        }
        else if (extension == "orc")
        {
            current_df = mod_cudf.attr("read_orc")(path.string());
        }
        else if (extension == "json")
        {
            current_df = mod_cudf.attr("read_json")(path.string());
        }
        else if (extension == "feather")
        {
            current_df = mod_cudf.attr("read_feather")(path.string());
        }
        else if (extension == "hdf")
        {
            current_df = mod_cudf.attr("read_hdf")(path.string());
        }
        else if (extension == "avro")
        {
            current_df = mod_cudf.attr("read_avro")(path.string());
        }

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
    }

    message->payload(MessageMeta::create_from_python(std::move(dataframe)));
    return message;
}
}  // namespace morpheus