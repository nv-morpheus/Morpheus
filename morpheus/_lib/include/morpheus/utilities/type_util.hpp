/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <neo/utils/type_utils.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <memory>
#include <stdexcept>

namespace morpheus {
/****** Component public implementations *******************/
/****** DType****************************************/
struct DType : neo::DataType  // NOLINT
{
    DType(const neo::DataType &dtype);

    DType(neo::TypeId tid);

    // Cudf representation
    cudf::type_id cudf_type_id() const;

    // Returns the triton string representation
    std::string triton_str() const;

    // from template
    template <typename T>
    static DType create()
    {
        return DType(neo::DataType::create<T>());
    }

    // From cudf
    static DType from_cudf(cudf::type_id tid);

    // From triton
    static DType from_triton(const std::string &type_str);
};

template <typename T>
DType type_to_dtype()
{
    return DType::from_triton(cudf::type_to_id<T>);
}

}  // namespace morpheus
