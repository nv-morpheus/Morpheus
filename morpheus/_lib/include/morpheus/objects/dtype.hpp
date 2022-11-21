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

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <climits>  // for CHAR_BIT
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>
#include <stdexcept>
#include <string>  // for string

namespace morpheus {
/****** Component public implementations *******************/

// Pulled from cudf
#pragma GCC visibility push(default)
enum class TypeId : int32_t
{
    EMPTY,    ///< Always null with no underlying data
    INT8,     ///< 1 byte signed integer
    INT16,    ///< 2 byte signed integer
    INT32,    ///< 4 byte signed integer
    INT64,    ///< 8 byte signed integer
    UINT8,    ///< 1 byte unsigned integer
    UINT16,   ///< 2 byte unsigned integer
    UINT32,   ///< 4 byte unsigned integer
    UINT64,   ///< 8 byte unsigned integer
    FLOAT32,  ///< 4 byte floating point
    FLOAT64,  ///< 8 byte floating point
    BOOL8,    ///< Boolean using one byte per value, 0 == false, else true
    STRING,   ///< String elements, not supported by cupy

    //   TIMESTAMP_DAYS,          ///< point in time in days since Unix Epoch in int32
    //   TIMESTAMP_SECONDS,       ///< point in time in seconds since Unix Epoch in int64
    //   TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in int64
    //   TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in int64
    //   TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in int64
    //   DURATION_DAYS,           ///< time interval of days in int32
    //   DURATION_SECONDS,        ///< time interval of seconds in int64
    //   DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
    //   DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
    //   DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
    //   DICTIONARY32,            ///< Dictionary type using int32 indices
    //   LIST,                    ///< List elements
    //   DECIMAL32,               ///< Fixed-point type with int32_t
    //   DECIMAL64,               ///< Fixed-point type with int64_t
    //   STRUCT,                  ///< Struct elements

    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS  ///< Total number of type ids
};

// Pulled from cuDF
template <typename T>
constexpr std::size_t size_in_bits()
{
    static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
    return sizeof(T) * CHAR_BIT;
}

/****** DType****************************************/
struct DType  // TODO move to dtype.hpp
{
    DType(TypeId tid);
    DType(const DType &dtype) = default;
    bool operator==(const DType &other) const;

    TypeId type_id() const;

    // Number of bytes per item
    size_t item_size() const;

    // Pretty print
    std::string name() const;

    // Returns the numpy string representation
    std::string type_str() const;

    // Cudf representation
    cudf::type_id cudf_type_id() const;

    // Returns the triton string representation
    std::string triton_str() const;

    // From cudf
    static DType from_cudf(cudf::type_id tid);

    // From numpy
    static DType from_numpy(const std::string &numpy_str);

    // From triton
    static DType from_triton(const std::string &type_str);

    // from template
    template <typename T>
    static DType create()
    {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 8)
        {
            return DType(TypeId::INT8);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 16)
        {
            return DType(TypeId::INT16);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 32)
        {
            return DType(TypeId::INT32);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 64)
        {
            return DType(TypeId::INT64);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 8)
        {
            return DType(TypeId::UINT8);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 16)
        {
            return DType(TypeId::UINT16);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 32)
        {
            return DType(TypeId::UINT32);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 64)
        {
            return DType(TypeId::UINT64);
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 32)
        {
            return DType(TypeId::FLOAT32);
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 64)
        {
            return DType(TypeId::FLOAT64);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return DType(TypeId::BOOL8);
        }
        else
        {
            static_assert(!sizeof(T), "Type not implemented");
        }

        // To hide compiler warnings
        return DType(TypeId::EMPTY);
    }

  private:
    char type_char() const;

    TypeId m_type_id;
};

template <typename T>
DType type_to_dtype()
{
    return DType::from_triton(cudf::type_to_id<T>);
}
#pragma GCC visibility pop
}  // namespace morpheus
