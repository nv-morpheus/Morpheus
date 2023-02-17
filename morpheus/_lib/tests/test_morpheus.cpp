/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "test_morpheus.hpp"

#include "morpheus/io/data_loader_registry.hpp"
#include "morpheus/io/loaders/all.hpp"
#include "morpheus/messages/meta.hpp"

#include <pybind11/pybind11.h>

#include <filesystem>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
std::string accum_merge(std::string lhs, std::string rhs)
{
    if (lhs.empty())
    {
        return std::move(rhs);
    }

    return std::move(lhs) + "," + std::move(rhs);
}
}  // namespace

namespace morpheus::test {

bool TestWithPythonInterpreter::m_initialized = false;

void TestWithPythonInterpreter::SetUp()
{
    initialize_interpreter();

    LoaderRegistry::register_factory_fn(
        "file", []() { return std::make_unique<FileDataLoader>(); }, false);
    LoaderRegistry::register_factory_fn(
        "grpc", []() { return std::make_unique<GRPCDataLoader>(); }, false);
    LoaderRegistry::register_factory_fn(
        "payload", []() { return std::make_unique<PayloadDataLoader>(); }, false);
    LoaderRegistry::register_factory_fn(
        "rest", []() { return std::make_unique<RESTDataLoader>(); }, false);
}

void TestWithPythonInterpreter::TearDown() {}

void TestWithPythonInterpreter::initialize_interpreter() const
{
    if (!m_initialized)
    {
        pybind11::initialize_interpreter();
        m_initialized = true;
    }
}

std::filesystem::path get_morpheus_root()
{
    auto root = std::getenv("MORPHEUS_ROOT");

    if (root == nullptr)
    {
        throw std::runtime_error("MORPHEUS_ROOT env variable is not set");
    }

    return std::filesystem::path{root};
}

std::string create_mock_csv_file(std::vector<std::string> cols, std::vector<std::string> dtypes, std::size_t rows)
{
    assert(cols.size() == dtypes.size());
    static std::vector<std::string> random_strings = {"field1", "test123", "abc", "xyz", "123", "foo", "bar", "baz"};

    auto sstream = std::stringstream();

    // Create header
    sstream << std::accumulate(cols.begin(), cols.end(), std::string(""), accum_merge);
    sstream << std::endl;

    // Populate with random data
    std::srand(std::time(nullptr));
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols.size(); ++col)
        {
            if (dtypes[col] == "int32")
            {
                sstream << std::rand() % 100 << ",";
            }
            else if (dtypes[col] == "float32")
            {
                sstream << std::rand() % 100 << "." << std::rand() % 100 << ",";
            }
            else if (dtypes[col] == "string")
            {
                sstream << random_strings[std::rand() % (random_strings.size() - 1)] << ",";
            }
            else
            {
                throw std::runtime_error(dtypes[col] + ": No");
            }
        }
        sstream.seekp(-1, std::ios::cur);  // Remove last comma
        sstream << std::endl;
    }

    return sstream.str();
}

std::shared_ptr<MessageMeta> create_mock_msg_meta(std::vector<std::string> cols,
                                                  std::vector<std::string> dtypes,
                                                  std::size_t rows)
{
    auto string_df = create_mock_csv_file(cols, dtypes, rows);

    pybind11::gil_scoped_acquire gil;
    pybind11::module_ mod_cudf;
    mod_cudf = pybind11::module_::import("cudf");

    auto py_string = pybind11::str(string_df);
    auto py_buffer = pybind11::buffer(pybind11::bytes(py_string));
    auto dataframe = mod_cudf.attr("read_csv")(py_buffer);

    return MessageMeta::create_from_python(std::move(dataframe));
}

}  // namespace morpheus::test
