/*
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

#include "test_morpheus.hpp"

#include "morpheus/objects/tensor_object.hpp"

#include <cuda/memory_resource>
#include <cuda_runtime.h>
#include <matx.h>
#include <matx_type_utils.h>
#include <mrc/cuda/common.hpp>  // for MRC_CHECK_CUDA
#include <mrc/cuda/sync.hpp>    // for enqueue_stream_sync_event
#include <mrc/memory/adaptors.hpp>
#include <mrc/memory/buffer.hpp>
#include <mrc/memory/literals.hpp>
#include <mrc/memory/old_interface/memory.hpp>
#include <mrc/memory/resources/device/cuda_malloc_resource.hpp>
#include <mrc/memory/resources/host/pinned_memory_resource.hpp>
#include <mrc/memory/resources/logging_resource.hpp>
#include <mrc/ucx/context.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

#include <algorithm>
#include <chrono>
#include <ratio>

using namespace mrc::memory::literals;
using namespace morpheus;

using RankType = int;

class TestCuda : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        MRC_CHECK_CUDA(cudaStreamCreate(&stream));

        auto pinned = std::make_shared<mrc::memory::pinned_memory_resource>();
        auto device = std::make_shared<mrc::memory::cuda_malloc_resource>(0);

        m_host_allocator   = mrc::memory::OldHostAllocator(pinned, nullptr).shared();
        m_device_allocator = mrc::memory::OldDeviceAllocator(device, nullptr).shared();
    }

    void TearDown() override
    {
        MRC_CHECK_CUDA(cudaStreamSynchronize(stream));
        MRC_CHECK_CUDA(cudaStreamDestroy(stream));
    }

    template <typename T, RankType R>
    TensorObject make_host_tensor(const TensorIndex (&shape)[R])
    {
        auto count = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
        auto md    = m_host_allocator->allocate_descriptor(count * sizeof(T)).make_shared();
        std::vector s(std::begin(shape), std::end(shape));

        auto tensor = std::make_shared<GenericTensor>(
            md, 0, DataType(TypeId::FLOAT32), std::vector<TensorIndex>{s}, std::vector<TensorIndex>{});

        return TensorObject(md, tensor);
    }

    template <typename T, RankType R>
    TensorObject make_device_tensor(const TensorIndex (&shape)[R])
    {
        auto count = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
        auto md    = m_device_allocator->allocate_descriptor(count * sizeof(T)).make_shared();
        std::vector s(std::begin(shape), std::end(shape));

        auto tensor = std::make_shared<GenericTensor>(md, 0, DataType(TypeId::FLOAT32), s);

        return TensorObject(md, std::move(tensor));
    }

    cudaStream_t stream;  // NOLINT

    std::shared_ptr<mrc::memory::IAllocator> m_host_allocator;
    std::shared_ptr<mrc::memory::IAllocator> m_device_allocator;
};

template <typename T>
auto await_matx(matx::BaseOp<T>& op, cudaStream_t stream)
{
    op.run(stream);
    return mrc::enqueue_stream_sync_event(stream);
}

void test_1d(const TensorObject& one_d)
{
    CHECK_EQ(one_d.rank(), 1);
    CHECK_EQ(one_d.dtype_size(), 4);
    CHECK_EQ(one_d.count(), 100);
    CHECK_EQ(one_d.bytes(), 400);
    CHECK_EQ(one_d.shape(0), 100);
    CHECK_EQ(one_d.stride(0), 1);
}

void test_2d(const TensorObject& two_d)
{
    CHECK_EQ(two_d.rank(), 2);
    CHECK_EQ(two_d.dtype_size(), 4);
    CHECK_EQ(two_d.count(), 100);
    CHECK_EQ(two_d.bytes(), 400);
    CHECK_EQ(two_d.shape(0), 10);
    CHECK_EQ(two_d.shape(1), 10);

    // row major
    CHECK_EQ(two_d.stride(0), 10);
    CHECK_EQ(two_d.stride(1), 1);
}

TEST_F(TestCuda, Tensor1D)
{
    auto one_d = make_host_tensor<float>({100});
    test_1d(one_d);

    auto two_d = one_d.reshape({10, 10});
    test_2d(two_d);
}

TEST_F(TestCuda, Tensor2D)
{
    auto two_d = make_host_tensor<float>({10, 10});
    test_2d(two_d);

    auto one_d = two_d.reshape({100});
    test_1d(one_d);

    CHECK_EQ(one_d.data(), two_d.data());
}

TEST_F(TestCuda, Shape)
{
    std::array<matx::index_t, 2> array_2d = {3, 5};
    matx::tensorShape_t<2> shape_2d(array_2d.data());
}
