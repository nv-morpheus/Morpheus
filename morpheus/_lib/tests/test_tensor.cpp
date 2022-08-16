/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/objects/tensor_object.hpp"  // for TensorIndex
#include "morpheus/utilities/tensor_util.hpp"  // for TensorUtils, TensorUtils::shape_type

#include <gtest/gtest.h>  // for AssertionResult, SuiteApiResolver, TestInfo, EXPECT_TRUE, Message, TEST_F, Test, TestFactoryImpl, TestPartResult

#include <cstddef>  // for size_t
#include <string>   // for allocator, operator==, basic_string, string
#include <vector>   // for vector
// IWYU pragma: no_include "morpheus/utilities/string_util.hpp"
// IWYU thinks we need ext/new_allocator.h for size_t for some reason
// IWYU pragma: no_include <ext/new_allocator.h>

using namespace morpheus;

class TestTensor : public ::testing::Test
{
  protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TestTensor, UtilsShapeString)
{
    TensorUtils::shape_type shape = {100, 10, 1};
    auto shape_str                = TensorUtils::shape_to_string(shape);
    EXPECT_TRUE(shape_str == std::string("(100, 10, 1)"));
}

TEST_F(TestTensor, GetElementStride)
{
    EXPECT_EQ(TensorUtils::get_element_stride<TensorIndex>({10, 1}), TensorUtils::shape_type({10, 1}));
    EXPECT_EQ(TensorUtils::get_element_stride<TensorIndex>({1, 13}), TensorUtils::shape_type({1, 13}));
    EXPECT_EQ(TensorUtils::get_element_stride<TensorIndex>({8, 104}), TensorUtils::shape_type({1, 13}));
    EXPECT_EQ(TensorUtils::get_element_stride<TensorIndex>({8, 16, 112}), TensorUtils::shape_type({1, 2, 14}));

    EXPECT_EQ(TensorUtils::get_element_stride<std::size_t>({10, 1}), std::vector<std::size_t>({10, 1}));
    EXPECT_EQ(TensorUtils::get_element_stride<std::size_t>({1, 13}), std::vector<std::size_t>({1, 13}));
    EXPECT_EQ(TensorUtils::get_element_stride<std::size_t>({8, 104}), std::vector<std::size_t>({1, 13}));
    EXPECT_EQ(TensorUtils::get_element_stride<std::size_t>({8, 16, 112}), std::vector<std::size_t>({1, 2, 14}));

    {
        auto results = TensorUtils::get_element_stride<TensorIndex, std::size_t>({10, 1});
        EXPECT_EQ(results, TensorUtils::shape_type({10, 1}));
    }

    {
        auto results = TensorUtils::get_element_stride<TensorIndex, std::size_t>({1, 13});
        EXPECT_EQ(results, TensorUtils::shape_type({1, 13}));
    }

    {
        auto results = TensorUtils::get_element_stride<TensorIndex, std::size_t>({8, 104});
        EXPECT_EQ(results, TensorUtils::shape_type({1, 13}));
    }

    {
        auto results = TensorUtils::get_element_stride<TensorIndex, std::size_t>({8, 16, 112});
        EXPECT_EQ(results, TensorUtils::shape_type({1, 2, 14}));
    }
}

/*
TEST_F(TestTensor, UtilsValidateShapeAndStride)
{
    // validate shape and stride works off element count without knowledge
    // the dtype size
    //
    // stride 1 tensors must have a sorted_index(shape).begin() == 1

    void *ptr         = reinterpret_cast<void *>(0xDEADBEEF);
    std::size_t bytes = 32 * 1024 * 1024;  // 32 MB

    memory::blob mv(ptr, bytes, memory::memory_kind_type::pinned);

    TensorView t0(mv, DataType::create<float>(), {3, 320, 320});

    EXPECT_TRUE(TensorUtils::has_contiguous_stride(t0.shape(), t0.stride()));
    EXPECT_TRUE(TensorUtils::validate_shape_and_stride(t0.shape(), t0.stride()));

    EXPECT_EQ(t0.stride(), std::vector<TensorIndex>({320 * 320, 320, 1}));
}
*/
