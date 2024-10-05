/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // for get_morpheus_root, TEST_CLASS_WITH_PYTHON, morpheus

#include "morpheus/controllers/monitor_controller.hpp"  // for MonitorController
#include "morpheus/messages/control.hpp"                // for ControlMessage

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(MonitorController);

cudf::io::table_with_metadata create_cudf_table_with_metadata(int rows, int cols)
{
    std::vector<std::unique_ptr<cudf::column>> columns;

    for (int i = 0; i < cols; ++i)
    {
        auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, rows);
        auto col_view = col->mutable_view();

        std::vector<int32_t> data(rows);
        std::iota(data.begin(), data.end(), 0);
        cudaMemcpy(col_view.data<int32_t>(), data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

        columns.push_back(std::move(col));
    }

    auto table = std::make_unique<cudf::table>(std::move(columns));

    auto index_info   = cudf::io::column_name_info{""};
    auto column_names = std::vector<cudf::io::column_name_info>(cols, index_info);
    auto metadata     = cudf::io::table_metadata{std::move(column_names), {}, {}};

    return cudf::io::table_with_metadata{std::move(table), metadata};
}

TEST_F(TestMonitorController, TestAutoCountFn)
{
    auto message_meta_mc            = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta");
    auto message_meta_auto_count_fn = message_meta_mc.auto_count_fn();
    auto meta                       = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    EXPECT_EQ((*message_meta_auto_count_fn)(meta), 10);

    auto control_message_mc            = MonitorController<std::shared_ptr<ControlMessage>>("test_control_message");
    auto control_message_auto_count_fn = control_message_mc.auto_count_fn();
    auto control_message               = std::make_shared<ControlMessage>();
    auto cm_meta = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(20, 3)));
    control_message->payload(cm_meta);
    EXPECT_EQ((*control_message_auto_count_fn)(control_message), 20);

    auto message_meta_vector_mc =
        MonitorController<std::vector<std::shared_ptr<MessageMeta>>>("test_message_meta_vector");
    auto message_meta_vector_auto_count_fn = message_meta_vector_mc.auto_count_fn();
    std::vector<std::shared_ptr<MessageMeta>> meta_vector;
    for (int i = 0; i < 5; ++i)
    {
        meta_vector.emplace_back(MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(5, 2))));
    }
    EXPECT_EQ((*message_meta_vector_auto_count_fn)(meta_vector), 25);

    auto control_message_vector_mc =
        MonitorController<std::vector<std::shared_ptr<ControlMessage>>>("test_control_message_vector");
    auto control_message_vector_auto_count_fn = control_message_vector_mc.auto_count_fn();
    std::vector<std::shared_ptr<ControlMessage>> control_message_vector;
    for (int i = 0; i < 5; ++i)
    {
        auto cm = std::make_shared<ControlMessage>();
        cm->payload(MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(6, 2))));
        control_message_vector.emplace_back(cm);
    }
    EXPECT_EQ((*control_message_vector_auto_count_fn)(control_message_vector), 30);

    // Test invalid message type
    EXPECT_THROW(MonitorController<int>("invalid message type"), std::runtime_error);
}

TEST_F(TestMonitorController, TestProgressBar)
{
    auto message_meta_mc   = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta");
    auto meta              = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    auto message_meta_mc_2 = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta_2");
    auto meta_2            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    auto message_meta_mc_3 = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta_3");
    auto meta_3            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));

    auto control_message_mc = MonitorController<std::shared_ptr<ControlMessage>>("test_control_message");
    auto control_message    = std::make_shared<ControlMessage>();
    auto cm_meta            = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(20, 3)));
    control_message->payload(cm_meta);

    for (int i = 0; i < 1000; i++)
    {
        std::cout << "log message" << std::endl;
        // std::cout << "log message 2" << std::endl;
        // std::cout << "log message 3" << std::endl;
        message_meta_mc.progress_sink(meta);
        std::cout << "log message 2" << std::endl;
        message_meta_mc_2.progress_sink(meta_2);
        // std::cout << "log message 3" << std::endl;
        message_meta_mc_3.progress_sink(meta_3);
        control_message_mc.progress_sink(control_message);
        std::this_thread::sleep_until(std::chrono::system_clock::now() + std::chrono::milliseconds(1000));
    }
}
