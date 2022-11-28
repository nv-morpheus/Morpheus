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

#include "morpheus/utilities/type_util.hpp"

#include <stdexcept>
#include <string>

namespace morpheus {

DType::DType(const DataType& dtype) : DataType(dtype.type_id()) {}
DType::DType(TypeId tid) : DataType(tid) {}

// Cudf representation
cudf::type_id DType::cudf_type_id() const
{
    switch (m_type_id)
    {
    case TypeId::INT8:
        return cudf::type_id::INT8;
    case TypeId::INT16:
        return cudf::type_id::INT16;
    case TypeId::INT32:
        return cudf::type_id::INT32;
    case TypeId::INT64:
        return cudf::type_id::INT64;
    case TypeId::UINT8:
        return cudf::type_id::UINT8;
    case TypeId::UINT16:
        return cudf::type_id::UINT16;
    case TypeId::UINT32:
        return cudf::type_id::UINT32;
    case TypeId::UINT64:
        return cudf::type_id::UINT64;
    case TypeId::FLOAT32:
        return cudf::type_id::FLOAT32;
    case TypeId::FLOAT64:
        return cudf::type_id::FLOAT64;
    case TypeId::BOOL8:
        return cudf::type_id::BOOL8;
    case TypeId::EMPTY:
    case TypeId::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// Returns the triton string representation
std::string DType::triton_str() const
{
    // Triton doesn't have any definitions or enums. Wow
    switch (m_type_id)
    {
    case TypeId::INT8:
        return "INT8";
    case TypeId::INT16:
        return "INT16";
    case TypeId::INT32:
        return "INT32";
    case TypeId::INT64:
        return "INT64";
    case TypeId::UINT8:
        return "UINT8";
    case TypeId::UINT16:
        return "UINT16";
    case TypeId::UINT32:
        return "UINT32";
    case TypeId::UINT64:
        return "UINT64";
    case TypeId::FLOAT32:
        return "FP32";
    case TypeId::FLOAT64:
        return "FP64";
    case TypeId::BOOL8:
        return "BOOL";
    case TypeId::EMPTY:
    case TypeId::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// From cudf
DType DType::from_cudf(cudf::type_id tid)
{
    switch (tid)
    {
    case cudf::type_id::INT8:
        return DType(TypeId::INT8);
    case cudf::type_id::INT16:
        return DType(TypeId::INT16);
    case cudf::type_id::INT32:
        return DType(TypeId::INT32);
    case cudf::type_id::INT64:
        return DType(TypeId::INT64);
    case cudf::type_id::UINT8:
        return DType(TypeId::UINT8);
    case cudf::type_id::UINT16:
        return DType(TypeId::UINT16);
    case cudf::type_id::UINT32:
        return DType(TypeId::UINT32);
    case cudf::type_id::UINT64:
        return DType(TypeId::UINT64);
    case cudf::type_id::FLOAT32:
        return DType(TypeId::FLOAT32);
    case cudf::type_id::FLOAT64:
        return DType(TypeId::FLOAT64);
    case cudf::type_id::BOOL8:
        return DType(TypeId::BOOL8);
    case cudf::type_id::EMPTY:
    case cudf::type_id::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// From triton
DType DType::from_triton(const std::string& type_str)
{
    if (type_str == "INT8")
    {
        return DType(TypeId::INT8);
    }
    else if (type_str == "INT16")
    {
        return DType(TypeId::INT16);
    }
    else if (type_str == "INT32")
    {
        return DType(TypeId::INT32);
    }
    else if (type_str == "INT64")
    {
        return DType(TypeId::INT64);
    }
    else if (type_str == "UINT8")
    {
        return DType(TypeId::UINT8);
    }
    else if (type_str == "UINT16")
    {
        return DType(TypeId::UINT16);
    }
    else if (type_str == "UINT32")
    {
        return DType(TypeId::UINT32);
    }
    else if (type_str == "UINT64")
    {
        return DType(TypeId::UINT64);
    }
    else if (type_str == "FP32")
    {
        return DType(TypeId::FLOAT32);
    }
    else if (type_str == "FP64")
    {
        return DType(TypeId::FLOAT64);
    }
    else if (type_str == "BOOL")
    {
        return DType(TypeId::BOOL8);
    }
    else
    {
        throw std::runtime_error("Not supported");
    }
}

}  // namespace morpheus
