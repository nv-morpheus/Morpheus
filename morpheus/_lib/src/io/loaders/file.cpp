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

#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pymrc/utilities/object_cache.hpp>

#include <fstream>
#include <memory>

namespace {}

namespace morpheus {
std::shared_ptr<MessageMeta> FileDataLoader::load(MessageControl& message)
{
    VLOG(30) << "Called FileDataLoader::load()";

    // TODO(Devin) : error checking + improve robustness
    auto filenames = message.message()["files"];
    auto sstream   = std::stringstream();
    for (auto& filename : filenames)
    {
        auto file = std::fstream(filename);
        if (!file)
        {
            throw std::runtime_error("Could not open file: ");
        }

        // TODO(Devin) : implement strategies
        sstream << file.rdbuf();
        file.close();
    }

    {
        pybind11::gil_scoped_acquire gil;
        pybind11::module_ mod_cudf;

        auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
        mod_cudf           = cache_handle.get_module("cudf");

        // TODO(Devin) : Do something more efficient
        auto py_string = pybind11::str(sstream.str());
        auto py_buffer = pybind11::buffer(pybind11::bytes(py_string));
        auto dataframe = mod_cudf.attr("read_csv")(py_buffer);

        return MessageMeta::create_from_python(std::move(dataframe));
    }
}
}  // namespace morpheus