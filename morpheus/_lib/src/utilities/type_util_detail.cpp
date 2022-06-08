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

#include <morpheus/utilities/type_util_detail.hpp>

#include <morpheus/utilities/string_util.hpp>

#include <glog/logging.h>

#include <cstddef>  // for size_t
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>  // for pair

namespace morpheus {

std::map<char, std::map<size_t, TypeId>> make_str_to_type_id()
{
    std::map<char, std::map<size_t, TypeId>> map;

    map['?'][1] = TypeId::BOOL8;

    map['i'][1] = TypeId::INT8;
    map['i'][2] = TypeId::INT16;
    map['i'][4] = TypeId::INT32;
    map['i'][8] = TypeId::INT64;

    map['u'][1] = TypeId::UINT8;
    map['u'][2] = TypeId::UINT16;
    map['u'][4] = TypeId::UINT32;
    map['u'][8] = TypeId::UINT64;

    map['f'][4] = TypeId::FLOAT32;
    map['f'][8] = TypeId::FLOAT64;

    return map;
}

std::map<char, std::map<size_t, TypeId>> str_to_type_id = make_str_to_type_id();

DataType::DataType(TypeId tid) : m_type_id(tid) {}

TypeId DataType::type_id() const
{
    return m_type_id;
}

size_t DataType::item_size() const
{
    switch (m_type_id)
    {
    case TypeId::INT8:
    case TypeId::UINT8:
    case TypeId::BOOL8:
        return 1;
    case TypeId::INT16:
    case TypeId::UINT16:
        return 2;
    case TypeId::INT32:
    case TypeId::UINT32:
    case TypeId::FLOAT32:
        return 4;
    case TypeId::INT64:
    case TypeId::UINT64:
    case TypeId::FLOAT64:
        return 8;
    case TypeId::NUM_TYPE_IDS:
    case TypeId::EMPTY:
    default:
        throw std::invalid_argument("Unknown datatype");
    }
}

std::string DataType::name() const
{
    // TODO(MDD): Replace this with a better version. For now, follow type_str
    return this->type_str();
}

std::string DataType::type_str() const
{
    return MORPHEUS_CONCAT_STR("<" << this->type_char() << this->item_size());
}

bool DataType::operator==(const DataType& other) const
{
    return m_type_id == other.m_type_id;
}

DataType DataType::from_numpy(const std::string& numpy_str)
{
    CHECK(!numpy_str.empty()) << "Cannot create DataType from empty string";

    char type_char    = numpy_str[0];
    size_t size_start = 1;

    // Can start with < or > or none
    if (numpy_str[0] == '<' || numpy_str[0] == '>')
    {
        type_char  = numpy_str[1];
        size_start = 2;
    }

    auto dtype_size = std::stoi(numpy_str.substr(size_start));

    // Now lookup in the map
    auto found_type = str_to_type_id.find(type_char);

    CHECK(found_type != str_to_type_id.end()) << "Type char '" << type_char << "' not supported";

    auto found_enum = found_type->second.find(dtype_size);

    CHECK(found_enum != found_type->second.end()) << "Type str '" << type_char << dtype_size << "' not supported";

    return DataType(found_enum->second);
}

char DataType::type_char() const
{
    switch (m_type_id)
    {
    case TypeId::INT8:
    case TypeId::INT16:
    case TypeId::INT32:
    case TypeId::INT64:
        return 'i';
    case TypeId::UINT8:
    case TypeId::UINT16:
    case TypeId::UINT32:
    case TypeId::UINT64:
        return 'u';
    case TypeId::BOOL8:
        return '?';
    case TypeId::FLOAT32:
    case TypeId::FLOAT64:
        return 'f';
    case TypeId::NUM_TYPE_IDS:
    case TypeId::EMPTY:
    default:
        throw std::invalid_argument("Unknown datatype");
    }
}
}  // namespace morpheus
