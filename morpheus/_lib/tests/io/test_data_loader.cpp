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

#include "test_io.hpp"

#include "morpheus/io/data_loader.hpp"
#include "morpheus/io/loaders/all.hpp"
#include "morpheus/messages/control.hpp"

#include <mrc/core/executor.hpp>
#include <mrc/engine/pipeline/ipipeline.hpp>
#include <mrc/modules/segment_modules.hpp>
#include <mrc/node/rx_sink.hpp>
#include <mrc/node/rx_source.hpp>
#include <mrc/options/options.hpp>
#include <mrc/options/topology.hpp>
#include <mrc/pipeline/pipeline.hpp>
#include <mrc/segment/builder.hpp>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace morpheus;
using namespace morpheus::test;

bool TestIO::m_initialized{false};

TEST_F(TestDataLoader, DataLoaderInitializationTest)
{
    auto data_loader = DataLoader();
}

TEST_F(TestDataLoader, DataLoaderRegisterLoaderTest)
{
    auto data_loader = DataLoader();

    nlohmann::json config;
    config["loader_id"] = "";

    std::vector<std::string> loaders = {"grpc", "payload", "rest"};
    for (auto& loader : loaders)
    {
        config["loader_id"] = loader;
        auto msg            = MessageControl(config);

        EXPECT_THROW(data_loader.load(msg), std::runtime_error);
    }

    // data_loader.register_loader("file", std::make_unique<FileDataLoader>());
    data_loader.register_loader("grpc", std::make_unique<GRPCDataLoader>());
    data_loader.register_loader("payload", std::make_unique<PayloadDataLoader>());
    data_loader.register_loader("rest", std::make_unique<RESTDataLoader>());

    for (auto& loader : loaders)
    {
        config["loader_id"] = loader;
        auto msg            = MessageControl(config);

        EXPECT_NO_THROW(data_loader.load(msg));
    }
}

/**
 * @brief Check that we can send a control message, with a raw data payload and load it correctly.
 */
TEST_F(TestDataLoader, PayloadLoaderTest)
{
    auto data_loader = DataLoader();
    data_loader.register_loader("payload", std::make_unique<PayloadDataLoader>());

    nlohmann::json config;
    config["loader_id"] = "payload";

    auto msg = MessageControl(config);

    auto mm = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);
    msg.payload(mm);

    auto mm2 = data_loader.load(msg);
    EXPECT_EQ(mm, mm2);
}

/**
 * @brief Check that we can send a control message, with a raw data payload and load it correctly.
 */
TEST_F(TestDataLoader, FileLoaderTest)
{
    auto data_loader = DataLoader();
    data_loader.register_loader("file", std::make_unique<FileDataLoader>());

    auto string_df = create_mock_dataframe({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);

    char temp_file[] = "/tmp/morpheus_test_XXXXXXXX";
    int fd           = mkstemp(temp_file);
    if (fd == -1)
    {
        GTEST_SKIP() << "Failed to create temporary file, skipping test";
    }

    nlohmann::json config;
    config["loader_id"] = "file";
    config["strategy"]  = "merge";
    config["files"]     = {std::string(temp_file)};

    auto msg = MessageControl(config);

    std::fstream data_file(temp_file, std::ios::out | std::ios::binary | std::ios::trunc);
    data_file << string_df;
    data_file.close();

    auto mm2 = data_loader.load(msg);
}
