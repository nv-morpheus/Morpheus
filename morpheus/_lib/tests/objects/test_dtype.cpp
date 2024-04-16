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

#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/objects/dtype.hpp"  // for DType

#include <gtest/gtest.h>

#include <stdexcept>

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(DType);

TEST_F(TestDType, FromNumpyValidStr)
{
    DType dtype = DType::from_numpy("|i1");
    ASSERT_EQ(dtype.type_id(), TypeId::INT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|i1");

    dtype = DType::from_numpy("<i2");
    ASSERT_EQ(dtype.type_id(), TypeId::INT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<i2");

    dtype = DType::from_numpy("<i4");
    ASSERT_EQ(dtype.type_id(), TypeId::INT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<i4");

    dtype = DType::from_numpy("<i8");
    ASSERT_EQ(dtype.type_id(), TypeId::INT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<i8");

    dtype = DType::from_numpy("|u1");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|u1");

    dtype = DType::from_numpy("<u2");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<u2");

    dtype = DType::from_numpy("<u4");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<u4");

    dtype = DType::from_numpy("<u8");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<u8");

    dtype = DType::from_numpy("|b1");
    ASSERT_EQ(dtype.type_id(), TypeId::BOOL8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|b1");

    dtype = DType::from_numpy("<f4");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<f4");

    dtype = DType::from_numpy("<f8");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<f8");
}

TEST_F(TestDType, FromNumpyInValidStr)
{
    // invalid byte order char
    EXPECT_THROW(DType::from_numpy("{i2"), std::invalid_argument);

    // invalid type char
    EXPECT_THROW(DType::from_numpy("<x2"), std::invalid_argument);

    // invalid byte size
    EXPECT_THROW(DType::from_numpy("<i9"), std::invalid_argument);

    // all invalid
    EXPECT_THROW(DType::from_numpy("{x9"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("zzzzz"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("123456"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("*&()#%"), std::invalid_argument);

    // empty string
    EXPECT_THROW(DType::from_numpy(""), std::invalid_argument);
}