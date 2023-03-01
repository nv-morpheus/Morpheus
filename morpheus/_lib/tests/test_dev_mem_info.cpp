/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_morpheus.hpp"  // IWYU pragma: associated

#include "morpheus/objects/dev_mem_info.hpp"
#include "morpheus/objects/dtype.hpp"  // for DType
#include "morpheus/objects/memory_descriptor.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>  // for AssertionResult, SuiteApiResolver, TestInfo, EXPECT_TRUE, Message, TEST_F, Test, TestFactoryImpl, TestPartResult
#include <mrc/cuda/common.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>       // for arena_memory_resource
#include <rmm/mr/device/fixed_size_memory_resource.hpp>  // for fixed_size_memory_resource
#include <rmm/mr/device/owning_wrapper.hpp>              // for make_owning_wrapper
#include <rmm/mr/device/per_device_resource.hpp>
#include <sys/types.h>

#include <cstddef>  // for size_t
#include <memory>   // shared_ptr
#include <string>   // for allocator, operator==, basic_string, string
#include <vector>   // for vector

using namespace morpheus;

TEST_CLASS(DevMemInfo);

TEST_F(TestDevMemInfo, RmmBufferConstructor)
{
    const std::size_t rows = 20;
    const std::size_t cols = 5;
    auto dtype             = DType::create<float>();

    const auto byte_size = rows * cols * dtype.item_size();

    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = rmm::mr::make_owning_wrapper<rmm::mr::fixed_size_memory_resource>(
        std::make_shared<rmm::mr::cuda_memory_resource>());
    auto buffer = std::make_shared<rmm::device_buffer>(byte_size, rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer, dtype, {rows - 1, cols}, {1, rows}, dtype.item_size()};

    EXPECT_EQ(dm.bytes(), (rows - 1) * cols * dtype.item_size());
    EXPECT_EQ(dm.count(), (rows - 1) * cols);
    EXPECT_EQ(dm.offset_bytes(), dtype.item_size());

    EXPECT_EQ(dm.dtype(), dtype);
    EXPECT_EQ(dm.type_id(), dtype.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], rows - 1);
    EXPECT_EQ(dm.shape(0), rows - 1);
    EXPECT_EQ(dm.shape()[1], cols);
    EXPECT_EQ(dm.shape(1), cols);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], rows);
    EXPECT_EQ(dm.stride(1), rows);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + dtype.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource, mem_resource.get());
}

TEST_F(TestDevMemInfo, VoidPtrConstructor)
{
    const std::size_t rows = 20;
    const std::size_t cols = 5;
    auto dtype             = DType::create<float>();

    const auto byte_size = rows * cols * dtype.item_size();

    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource =
        rmm::mr::make_owning_wrapper<rmm::mr::arena_memory_resource>(std::make_shared<rmm::mr::cuda_memory_resource>());
    auto buffer = std::make_shared<rmm::device_buffer>(byte_size, rmm::cuda_stream_legacy, mem_resource.get());

    auto md = std::make_shared<MemoryDescriptor>(rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer->data(), dtype, md, {rows - 1, cols}, {1, rows}, dtype.item_size()};

    EXPECT_EQ(dm.bytes(), (rows - 1) * cols * dtype.item_size());
    EXPECT_EQ(dm.count(), (rows - 1) * cols);
    EXPECT_EQ(dm.offset_bytes(), dtype.item_size());

    EXPECT_EQ(dm.dtype(), dtype);
    EXPECT_EQ(dm.type_id(), dtype.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], rows - 1);
    EXPECT_EQ(dm.shape(0), rows - 1);
    EXPECT_EQ(dm.shape()[1], cols);
    EXPECT_EQ(dm.shape(1), cols);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], rows);
    EXPECT_EQ(dm.stride(1), rows);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + dtype.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource, mem_resource.get());
}
