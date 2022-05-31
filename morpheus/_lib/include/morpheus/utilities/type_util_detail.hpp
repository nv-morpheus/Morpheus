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

#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>

#ifndef NDEBUG
#include <cxxabi.h>
#endif

namespace morpheus {

// Pulled from cuDF
template <typename T>
constexpr std::size_t size_in_bits()
{
    static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
    return sizeof(T) * CHAR_BIT;
}

// Pulled from cudf
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
    //   STRING,                  ///< String elements
    //   LIST,                    ///< List elements
    //   DECIMAL32,               ///< Fixed-point type with int32_t
    //   DECIMAL64,               ///< Fixed-point type with int64_t
    //   STRUCT,                  ///< Struct elements

    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS  ///< Total number of type ids
};

struct DataType
{
    DataType(TypeId tid);

    TypeId type_id() const;

    // Number of bytes per item
    size_t item_size() const;

    // Pretty print
    std::string name() const;

    // Returns the numpy string representation
    std::string type_str() const;

    // // Returns the triton string representation
    // std::string triton_str() const;

    bool operator==(const DataType& other) const;

    // from template
    template <typename T>
    static DataType create()
    {
        if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 8)
        {
            return DataType(TypeId::INT8);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 16)
        {
            return DataType(TypeId::INT16);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 32)
        {
            return DataType(TypeId::INT32);
        }
        else if constexpr (std::is_integral_v<T> && std::is_signed_v<T> && size_in_bits<T>() == 64)
        {
            return DataType(TypeId::INT64);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 8)
        {
            return DataType(TypeId::UINT8);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 16)
        {
            return DataType(TypeId::UINT16);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 32)
        {
            return DataType(TypeId::UINT32);
        }
        else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T> && size_in_bits<T>() == 64)
        {
            return DataType(TypeId::UINT64);
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 32)
        {
            return DataType(TypeId::FLOAT32);
        }
        else if constexpr (std::is_floating_point_v<T> && size_in_bits<T>() == 64)
        {
            return DataType(TypeId::FLOAT64);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return DataType(TypeId::BOOL8);
        }
        else
        {
            static_assert(!sizeof(T), "Type not implemented");
        }

        // To hide compiler warnings
        return DataType(TypeId::EMPTY);
    }

    // From numpy
    static DataType from_numpy(const std::string& numpy_str);

  protected:
    char type_char() const;

    TypeId m_type_id;
};

}  // namespace morpheus
