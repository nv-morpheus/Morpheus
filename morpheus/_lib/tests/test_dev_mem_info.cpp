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

#include <gtest/gtest.h>  // for AssertionResult, SuiteApiResolver, TestInfo, EXPECT_TRUE, Message, TEST_F, Test, TestFactoryImpl, TestPartResult
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/arena_memory_resource.hpp>       // for arena_memory_resource
#include <rmm/mr/device/cuda_memory_resource.hpp>        // for cuda_memory_resource
#include <rmm/mr/device/fixed_size_memory_resource.hpp>  // for fixed_size_memory_resource
#include <rmm/mr/device/owning_wrapper.hpp>              // for make_owning_wrapper
#include <sys/types.h>

#include <cstddef>  // for size_t
#include <memory>   // shared_ptr
#include <vector>   // for vector
// IWYU pragma: no_include "thrust/iterator/iterator_facade.h"
// IWYU pragma: no_include <unordered_map>

using namespace morpheus;

namespace {
const std::size_t Rows = 20;
const std::size_t Cols = 5;
const auto Dtype       = DType::create<float>();

const auto ByteSize = Rows * Cols * Dtype.item_size();

template <template <typename> class ResourceT>
auto make_mem_resource()
{
    return rmm::mr::make_owning_wrapper<ResourceT>(std::make_shared<rmm::mr::cuda_memory_resource>());
}
}  // namespace

TEST_CLASS(DevMemInfo);

TEST_F(TestDevMemInfo, RmmBufferConstructor)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::fixed_size_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(ByteSize, rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer, Dtype, {Rows - 1, Cols}, {1, Rows}, Dtype.item_size()};

    EXPECT_EQ(dm.bytes(), (Rows - 1) * Cols * Dtype.item_size());
    EXPECT_EQ(dm.count(), (Rows - 1) * Cols);
    EXPECT_EQ(dm.offset_bytes(), Dtype.item_size());

    EXPECT_EQ(dm.dtype(), Dtype);
    EXPECT_EQ(dm.type_id(), Dtype.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], Rows - 1);
    EXPECT_EQ(dm.shape(0), Rows - 1);
    EXPECT_EQ(dm.shape()[1], Cols);
    EXPECT_EQ(dm.shape(1), Cols);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], Rows);
    EXPECT_EQ(dm.stride(1), Rows);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + Dtype.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource, mem_resource.get());
}

TEST_F(TestDevMemInfo, VoidPtrConstructor)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::arena_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(ByteSize, rmm::cuda_stream_legacy, mem_resource.get());

    auto md = std::make_shared<MemoryDescriptor>(rmm::cuda_stream_legacy, mem_resource.get());

    // Set the offset to the second row in the buffer
    DevMemInfo dm{buffer->data(), Dtype, md, {Rows - 1, Cols}, {1, Rows}, Dtype.item_size()};

    EXPECT_EQ(dm.bytes(), (Rows - 1) * Cols * Dtype.item_size());
    EXPECT_EQ(dm.count(), (Rows - 1) * Cols);
    EXPECT_EQ(dm.offset_bytes(), Dtype.item_size());

    EXPECT_EQ(dm.dtype(), Dtype);
    EXPECT_EQ(dm.type_id(), Dtype.type_id());

    EXPECT_EQ(dm.shape().size(), 2);
    EXPECT_EQ(dm.shape()[0], Rows - 1);
    EXPECT_EQ(dm.shape(0), Rows - 1);
    EXPECT_EQ(dm.shape()[1], Cols);
    EXPECT_EQ(dm.shape(1), Cols);

    EXPECT_EQ(dm.stride().size(), 2);
    EXPECT_EQ(dm.stride()[0], 1);
    EXPECT_EQ(dm.stride(0), 1);
    EXPECT_EQ(dm.stride()[1], Rows);
    EXPECT_EQ(dm.stride(1), Rows);

    EXPECT_EQ(dm.data(), static_cast<u_int8_t*>(buffer->data()) + Dtype.item_size());

    EXPECT_EQ(dm.memory()->cuda_stream, rmm::cuda_stream_legacy);
    EXPECT_EQ(dm.memory()->memory_resource, mem_resource.get());
}

TEST_F(TestDevMemInfo, MakeNewBuffer)
{
    // Explicitly passing the non-default values to make sure we don't have a test that accidentally passes
    auto mem_resource = make_mem_resource<rmm::mr::fixed_size_memory_resource>();
    auto buffer       = std::make_shared<rmm::device_buffer>(ByteSize, rmm::cuda_stream_legacy, mem_resource.get());

    DevMemInfo dm{buffer, Dtype, {Rows, Cols}, {1, Rows}};

    const std::size_t buff_size = 20;
    auto new_buff               = dm.make_new_buffer(buff_size);
    EXPECT_EQ(new_buff->size(), buff_size);
    EXPECT_EQ(new_buff->stream(), rmm::cuda_stream_legacy);
    EXPECT_EQ(new_buff->memory_resource(), mem_resource.get());
}
